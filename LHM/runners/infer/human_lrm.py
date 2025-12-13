# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

import os
from pathlib import Path

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


class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        print("\n--- Inference with config:")
        print(OmegaConf.to_yaml(self.cfg))

        self.facedetect = FaceDetector(
            "/scratch/izar/cizinsky/pretrained/pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )

        print(f"\n--- Loading Large Human Model")
        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)

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
        motion_seqs_dir,
    ):

        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        print(f"[DEBUG] motion_img_need_mask: {motion_img_need_mask}, vis_motion: {vis_motion}")

        img_np = cv2.imread(image_path)
        remove_np = remove(img_np)
        parsing_mask = remove_np[...,3]
        
        # prepare reference image
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

        # prepare head image
        print(f"[DEBUG] preparing head image for {image_path}")
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("[WARNING] w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)
        print(f"[DEBUG] saved refer image to {save_ref_img_path}")

        # save head image for vis
        save_head_img_path = os.path.join(
            dump_tmp_dir, "head_" + os.path.basename(image_path)
        )
        vis_head_img = (src_head_rgb[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_head_img).save(save_head_img_path)
        print(f"[DEBUG] saved head image to {save_head_img_path}")

        # Infer canonical gs model and query points
        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        # read motion seq
#         motion_seq = prepare_motion_seqs(
            # motion_seqs_dir,
            # motion_img_dir,
            # save_root=dump_tmp_dir,
            # fps=motion_video_read_fps,
            # bg_color=1.0,
            # aspect_standard=aspect_standard,
            # enlarge_ratio=[1.0, 1, 0],
            # render_image_res=render_size,
            # multiply=16,
            # need_mask=motion_img_need_mask,
            # vis_motion=vis_motion,
        # )
        motion_seq = prepare_motion_seqs_human3r(Path(motion_seqs_dir))

        # Save motion seq
        motion_seq_save_path = Path(self.cfg.save_dir) / f"motion_seq.pt"
        torch.save(motion_seq, motion_seq_save_path)
        print(f"[DEBUG] saved motion sequence to {motion_seq_save_path}")


        # Get canonical gs model based on the predicted betas
        self.model.to(dtype)
        gs_model_list, _, _ = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            smplx_params={"betas": shape_param.to(device)}
        )
        print(f"Shape of the shape params: {shape_param.shape}")
        gs_model_save_path = Path(self.cfg.save_dir) / "gs_model_list.pt"
        torch.save(gs_model_list, gs_model_save_path)


#        # Get canonical gs model based on the gt betas
        #smplx_dir = Path("/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17/smplx")
        #frame_paths = sorted([p for p in os.listdir(smplx_dir) if p.endswith(".npz")])
        #npzs = []
        #for fp in frame_paths:
            #npz = np.load(smplx_dir / f"{Path(fp).stem}.npz")
            #npzs.append(npz)

        #def stack_key(key):
            #arrs = [torch.from_numpy(n[key]).float() for n in npzs]
            #return torch.stack(arrs, dim=1).to(device)  # [P, F, ...]

        #betas = stack_key("betas")[1:2, 0, :10] # [P, 10]

        #gs_model_list, _, _ = self.model.infer_single_view(
            #image.unsqueeze(0).to(device, dtype),
            #src_head_rgb.unsqueeze(0).to(device, dtype),
            #smplx_params={"betas": betas.to(device)}
        #)
        #gs_model_save_path = Path(self.cfg.save_dir) / "gt_gs_model_list.pt"
        #torch.save(gs_model_list, gs_model_save_path)

        return 


    def infer(self):

        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            suffixes = (".jpg", ".jpeg", ".png", ".webp", ".JPG")
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith(suffixes):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        print(f"[DEBUG] total {len(image_paths)} images to process. Path to the first image: {image_paths[0]}")




        for image_path in tqdm(image_paths):
            self.infer_single(image_path, motion_seqs_dir=self.cfg.motion_seqs_dir)
