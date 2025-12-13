# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

import os
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox


from rembg import remove

from LHM.models.modeling_human_lrm import ModelHumanLRM
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    prepare_motion_seqs_human3r,
    resize_image_keepaspect_np,
)
from LHM.utils.face_detector import FaceDetector
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger

from .base_inferrer import Inferrer

def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded


def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)
    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box


def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr

def save_image(image: torch.Tensor, image_path: str):
    image_to_save = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
        np.uint8
    )
    Image.fromarray(image_to_save).save(image_path)

class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def __init__(self, cfg):
        super().__init__()

        # Config setup
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.save_dir = self.output_dir / "canon_3dgs_lhm"
        os.makedirs(self.save_dir, exist_ok=True)
        print("\n--- Inference with config:")
        print(OmegaConf.to_yaml(self.cfg))

        # Input data loading
        self._load_inference_inputs()

        # Model loading
        print(f"\n--- Loading Models")
        self.facedetect = FaceDetector(
            "/scratch/izar/cizinsky/pretrained/pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)


    def _load_inference_inputs(self):
        # Load inputs for the inference
        # - Masked images
        masked_images_dir = self.output_dir / "masked_images"
        person_ids = sorted([pid for pid in os.listdir(masked_images_dir) if pid.isdigit()])
        self.image_paths = []
        for pid in person_ids:
            image_path = masked_images_dir / pid / f"{self.cfg.input_image_id:04d}.png"
            assert image_path.exists(), f"Image path does not exist: {image_path}"
            self.image_paths.append(image_path)

        # - Shape params
        # -- Estimated from human3r
        shape_params_dir = self.output_dir / "motion_human3r"
        self.estimated_shape_params = []
        for pid in person_ids:
            shaper_params_path = shape_params_dir / pid / "smplx_params" / "00000.json"
            assert shaper_params_path.exists(), f"Shape params path does not exist: {shaper_params_path}"
            with open(shaper_params_path, 'r') as f:
                smplx_params = json.load(f)
                betas = np.array(smplx_params['betas']) # [10,]
                self.estimated_shape_params.append(betas)
        assert len(self.image_paths) == len(self.estimated_shape_params), "Number of images and shape params do not match"

        # -- Hi4D GT shape params
        if self.cfg.hi4d_gt_root_dir is not None:
            gt_smplx_params_dir = Path(self.cfg.hi4d_gt_root_dir) / "smplx"
            gt_smplx_params_path = gt_smplx_params_dir / "000001.npz"
            params = np.load(gt_smplx_params_path)
            betas = params['betas'] # [2, 10]
            self.hi4d_gt_shape_params = [betas[i] for i in range(betas.shape[0])]
            assert len(self.image_paths) == len(self.hi4d_gt_shape_params), "Number of images and shape params do not match"
        else:
            self.hi4d_gt_shape_params = list()

    def _build_model(self, cfg):
        from LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])

        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def crop_face_image(self, image_path):
        rgb = np.array(Image.open(image_path))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox = self.facedetect(rgb)
        head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        head_rgb = head_rgb.permute(1, 2, 0)
        head_rgb = head_rgb.cpu().numpy()
        return head_rgb

    def infer_single(
        self,
        image_path: str,
        shape_param: np.ndarray,
        shape_params_source: str = "estimated",
        person_id: int = 0,
    ):
        # create save dir
        save_dir_root = self.save_dir / f"{person_id:02d}"
        os.makedirs(save_dir_root, exist_ok=True)

        # prepare input image
        # - preprocess
        img_np = cv2.imread(image_path)
        remove_np = remove(img_np)
        parsing_mask = remove_np[...,3] 
        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        # - visualise and save
        image_save_path = save_dir_root / f"{shape_params_source}_input_image.png"
        save_image(image, image_save_path)

        # prepare head image
        # - preprocess
        src_head_rgb = self.crop_face_image(image_path)
        src_head_rgb = cv2.resize(
            src_head_rgb,
            dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
            interpolation=cv2.INTER_AREA,
        )  # resize to dino size
        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # - visualise and save
        head_image_save_path = save_dir_root / f"{shape_params_source}_input_head.png"
        save_image(src_head_rgb, head_image_save_path)


        # get canonical gs model based on the predicted betas
        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)
        self.model.to(dtype)
        gs_model_list, _, _ = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            smplx_params={"betas": shape_param.to(device)}
        )
        gs_model_save_path = save_dir_root / f"{shape_params_source}_gs.pt"
        torch.save(gs_model_list[0], gs_model_save_path)

    def infer(self):
        
        # Inference per person
        n_images = len(self.image_paths)
        estimated_canon_3dgs = []
        hi4d_gt_canon_3dgs = []
        for pid in range(n_images):
            print(f"\n--- Inference for person {pid+1} / {n_images}")
            person_image_path = self.image_paths[pid]
            person_estimated_shape_params = self.estimated_shape_params[pid]
            canon_gs = self.infer_single(person_image_path, person_estimated_shape_params, "human3r", pid)
            estimated_canon_3dgs.append(canon_gs)

            if len(self.hi4d_gt_shape_params) > 0:
                person_gt_shape_params = self.hi4d_gt_shape_params[pid]
                canon_gs = self.infer_single(person_image_path, person_gt_shape_params, "hi4d", pid)
                hi4d_gt_canon_3dgs.append(canon_gs)
        
        # Joined the outputs per person into a single dir
        save_dir_root = self.save_dir / f"union"
        os.makedirs(save_dir_root, exist_ok=True)
        torch.save(estimated_canon_3dgs, save_dir_root / "human3r_gs.pt")

        if len(hi4d_gt_canon_3dgs) > 0:
            torch.save(hi4d_gt_canon_3dgs, save_dir_root / "hi4d_gs.pt")