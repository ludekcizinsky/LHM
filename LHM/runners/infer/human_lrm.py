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
        self.output_dir = Path(cfg.scene_dir)
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

        # Get person ids based on individual masks
        images_root = self.output_dir / "images"
        cam_dirs = [p for p in images_root.iterdir() if p.is_dir()]
        if len(cam_dirs) != 1:
            raise ValueError(
                f"Expected exactly one camera folder under {images_root}, found {len(cam_dirs)}"
            )
        cam_id = cam_dirs[0].name

        seg_root = self.output_dir / "seg" / "img_seg_mask" / cam_id
        if not seg_root.exists():
            raise FileNotFoundError(f"Segmentation folder not found: {seg_root}")

        person_ids = []
        for p in seg_root.iterdir():
            if not p.is_dir():
                continue
            if p.name == "all":
                continue
            if p.name.isdigit():
                person_ids.append(int(p.name))
        person_ids = sorted(person_ids)
        if len(person_ids) == 0:
            raise ValueError(f"No per-person masks found under {seg_root}")

        # Resolve the reference frame by index into the sorted image list
        image_dir = cam_dirs[0]
        image_paths = list(image_dir.glob("*.jpg"))
        if not image_paths:
            image_paths = list(image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No image frames found in {image_dir}")
        try:
            image_paths.sort(key=lambda p: int(p.stem))
        except Exception:
            image_paths.sort()

        if self.cfg.input_image_idx < 0 or self.cfg.input_image_idx >= len(image_paths):
            raise IndexError(
                f"input_image_idx={self.cfg.input_image_idx} is out of range for {len(image_paths)} frames"
            )
        frame_path = image_paths[self.cfg.input_image_idx]
        frame_stem = frame_path.stem

        # Load inputs for the inference
        # - Masked images
        self.image_paths = []
        lhm_dir = self.output_dir / "misc" / "lhm"
        lhm_dir.mkdir(parents=True, exist_ok=True)

        rgb = cv2.imread(str(frame_path))
        if rgb is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")

        for pid in person_ids:
            mask_path = seg_root / f"{pid}" / f"{frame_stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_bool = mask > 0
            masked_rgb = np.zeros_like(rgb)
            masked_rgb[mask_bool] = rgb[mask_bool]

            masked_path = lhm_dir / f"person_{pid:02d}_{frame_stem}.png"
            cv2.imwrite(str(masked_path), masked_rgb)
            self.image_paths.append(str(masked_path))

        # - Shape params
        self.estimated_shape_params = [] # one array per person, each array is [10,]
        smplx_path = self.output_dir / "smplx" / f"{frame_stem}.npz"
        if not smplx_path.exists():
            raise FileNotFoundError(f"SMPL-X file not found: {smplx_path}")
        with np.load(smplx_path) as smplx_data:
            if "betas" not in smplx_data:
                raise KeyError(f"'betas' not found in SMPL-X file: {smplx_path}")
            betas = smplx_data["betas"]

        for pid in person_ids:
            if pid >= betas.shape[0]:
                raise IndexError(
                    f"Person id {pid} out of range for betas shape {betas.shape} in {smplx_path}"
                )
            self.estimated_shape_params.append(betas[pid].astype(np.float32))


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
        image_save_path = save_dir_root / f"input_image.png"
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
        head_image_save_path = save_dir_root / f"input_head.png"
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
        gs_model_save_path = save_dir_root / f"gs.pt"
        torch.save(gs_model_list[0], gs_model_save_path)

        return gs_model_list[0]

    def infer(self):
        
        # Inference per person
        n_images = len(self.image_paths)
        estimated_canon_3dgs = []
        for pid in range(n_images):
            print(f"\n--- Inference for person {pid+1} / {n_images}")
            person_image_path = self.image_paths[pid]
            person_estimated_shape_params = self.estimated_shape_params[pid]
            canon_gs = self.infer_single(person_image_path, person_estimated_shape_params, pid)
            estimated_canon_3dgs.append(canon_gs)

        
        # Joined the outputs per person into a single dir
        save_dir_root = self.save_dir / f"union"
        os.makedirs(save_dir_root, exist_ok=True)
        torch.save(estimated_canon_3dgs, save_dir_root / "gs.pt")
