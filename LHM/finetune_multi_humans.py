"""
Finetune canonical Gaussian params for multiple humans using original frames as supervision.

Trains offset_xyz, rotation, scaling, opacity, and SH/RGB while keeping SMPL-X motion fixed.
Supervision:
  - L_rgb: MSE between rendered RGB (masked) and GT frame * union mask (weight 1.0)
  - L_sil: MSE between rendered mask and GT union mask (weight 0.5)
  - L_ssim: SSIM (weight configurable)

Expects:
  output_dir/initial_scene_recon/<track_id>/* (loaded canonical state)
  output_dir/frames/<0000.png...> and output_dir/masks/union/<0000.png...> for supervision.
Saves finetuned state to output_dir/refined_scene_recon/<track_id>/...
"""


import os
import sys
import shutil
import subprocess
import imageio.v2 as imageio
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple
import copy
from tqdm import tqdm

from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fused_ssim import fused_ssim
import matplotlib.cm as cm
import pyiqa
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LHM.models import ModelHumanLRMSapdinoBodyHeadSD3_5
from LHM.outputs.output import GaussianAppOutput
from LHM.runners.infer.base_inferrer import Inferrer
from LHM.utils.hf_hub import wrap_model_hub

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
import math

def load_camera_from_npz(
    camera_npz_path: str | Path, camera_id: int, device: torch.device | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load intrinsics and extrinsics for a specific camera ID from a .npz file.

    Expects keys "ids", "intrinsics" [N,3,3], and "extrinsics" [N,3,4] in the file.
    Returns float tensors (intrinsics, extrinsics) optionally moved to `device`.
    """
    camera_npz_path = Path(camera_npz_path)
    with np.load(camera_npz_path) as cams:
        missing = [k for k in ("ids", "intrinsics", "extrinsics") if k not in cams.files]
        if missing:
            raise KeyError(f"Missing keys {missing} in camera file {camera_npz_path}")

        ids = cams["ids"]
        matches = np.nonzero(ids == camera_id)[0]
        if len(matches) == 0:
            raise ValueError(f"Camera id {camera_id} not found in {camera_npz_path}")
        idx = int(matches[0])

        intrinsics = torch.from_numpy(cams["intrinsics"][idx]).float()
        extrinsics = torch.from_numpy(cams["extrinsics"][idx]).float()

    if device is not None:
        device = torch.device(device)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)

    return intrinsics, extrinsics

def save_image(tensor: torch.Tensor, filename: str):
    """
    Accepts HWC, CHW, or BCHW; if batch > 1, saves the first item.
    Assumes values in [0,1]. If channels are a multiple of 3, tiles them along width.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] % 3 == 0:
        c = tensor.shape[0]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(n, 3, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor[i] for i in range(n)], dim=2)
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    elif tensor.dim() == 3 and tensor.shape[-1] % 3 == 0:
        c = tensor.shape[-1]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], n, 3)
            tensor = torch.cat([tensor[:, :, i] for i in range(n)], axis=1)
        image = tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for save_image: {tensor.shape}")
    image = (image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(image).save(filename)

def compute_frame_centers_from_smplx(smplx_params: dict) -> torch.Tensor:
    """
    smplx_params: dict containing key "trans" of shape [num_people, T, 3]
    returns: centers of shape [1, T, 3] (mean of all people per frame)
    """
    trans = smplx_params["trans"]  # [num_people, T, 3]
    centers = trans.mean(dim=0, keepdim=True)  # [1, T, 3]
    return centers

def rotate_c2ws_y_about_center(c2ws: torch.Tensor, centers: torch.Tensor, degrees: float) -> torch.Tensor:
    """
    Yaw cameras around a per-frame center on the world Y-axis.
    c2ws: [..., 4, 4]
    centers: [..., 3] matching leading dims of c2ws
    returns: same shape as c2ws
    """
    # Ensure dtype/device alignment
    centers = centers.to(dtype=c2ws.dtype, device=c2ws.device)

    rad = math.radians(-degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    R = torch.tensor(
        [[cos, 0.0, sin, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-sin, 0.0, cos, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=c2ws.dtype,
        device=c2ws.device,
    )

    # Broadcast R to leading dims
    while R.dim() < c2ws.dim():
        R = R.unsqueeze(0)

    # Build T(-center) and T(center) with shapes matching c2ws leading dims.
    leading_shape = c2ws.shape[:-2]
    I = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device)
    T_neg = I.expand(*leading_shape, 4, 4).clone()
    T_pos = I.expand(*leading_shape, 4, 4).clone()
    T_neg[..., :3, 3] = -centers
    T_pos[..., :3, 3] = centers

    # Apply: T(center) @ R_y @ T(-center) @ c2w
    return T_pos @ (R @ (T_neg @ c2ws))


def depth_to_color(
    arr: np.ndarray, mask: np.ndarray | None = None, vmin: float | None = None, vmax: float | None = None
) -> np.ndarray:
    """Normalize depth array to 0-1 (optionally over masked region and shared vmin/vmax) and apply magma colormap."""
    arr = arr.copy()
    if mask is not None:
        arr = arr * (mask > 0.5)
        vals = arr[mask > 0.5]
    else:
        vals = arr
    vals = vals[np.isfinite(vals)]
    if vmin is None or vmax is None:
        if vals.size > 0:
            vmin = vals.min()
            vmax = vals.max()
        else:
            vmin, vmax = 0.0, 1.0
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin)
    else:
        norm = arr * 0.0
    norm = np.clip(norm, 0, 1)
    return (cm.magma(norm)[..., :3] * 255).astype(np.uint8)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FrameMaskDataset(Dataset):
    def __init__(self, frames_dir: Path, masks_dir: Path, device: torch.device, sample_every: int = 1):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.device = device

        frame_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            frame_candidates.extend(frames_dir.glob(ext))
        self.frame_paths = sorted(set(frame_candidates))
        if sample_every > 1:
            self.frame_paths = self.frame_paths[::sample_every]
        if not self.frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}")
        self.mask_paths = []
        missing = []
        for p in self.frame_paths:
            base = p.stem
            candidates = [masks_dir / f"{base}{ext}" for ext in (".png", ".jpg", ".jpeg")]
            mask_path = next((c for c in candidates if c.exists()), None)
            if mask_path is None:
                missing.append(base)
            else:
                self.mask_paths.append(mask_path)
        if missing:
            raise RuntimeError(f"Missing masks for frames (by stem): {missing[:5]}")

    def __len__(self):
        return len(self.frame_paths)

    def _load_img(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.to(self.device)

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        arr = torch.from_numpy(np.array(mask)).float() / 255.0
        return arr.unsqueeze(-1).to(self.device)

    def __getitem__(self, idx: int):
        frame = self._load_img(self.frame_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        return (
            torch.tensor(idx, device=self.device, dtype=torch.long),
            frame,
            mask,
            str(self.frame_paths[idx]),
        )


# ---------------------------------------------------------------------------
# Utility to make Gaussian params trainable
# ---------------------------------------------------------------------------
def enable_gaussian_grads(
    gauss: GaussianAppOutput,
    train_fields: Tuple[str, ...],
    detach_to_leaf: bool = False,
):
    for f in fields(gauss):
        if f.name not in train_fields:
            continue
        v = getattr(gauss, f.name)
        if torch.is_tensor(v):
            if detach_to_leaf:
                v = v.detach().requires_grad_()
                setattr(gauss, f.name, v)
            else:
                v.requires_grad_(True)


# ---------------------------------------------------------------------------
# Finetuner
# ---------------------------------------------------------------------------
class MultiHumanFinetuner(Inferrer):
    EXP_TYPE = "multi_human_finetune"

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tuner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.output_dir).expanduser()
        self.train_params = tuple(cfg.train_params)
        self.wandb_run = None
        self.loaded_skinning = None

        self._load_gs_model(self.output_dir)
        self._prepare_joined_inputs()
        self.model: ModelHumanLRMSapdinoBodyHeadSD3_5 = self._build_model().to(self.tuner_device)
        self._apply_loaded_skinning()

        self.frames_dir = self.output_dir / "frames"
        self.masks_dir = self.output_dir / "masks" / "union"
        self.depth_dir = self.output_dir / "depth_maps" / "raw"

#        # clean this dir
        #save_root = self.output_dir / "refined_scene_recon" / self.cfg.exp_name
        #if save_root.exists():
            #shutil.rmtree(save_root)

    # ---------------- Model / data loading ----------------
    def _build_model(self):
        hf_model_cls = wrap_model_hub(ModelHumanLRMSapdinoBodyHeadSD3_5)
        model_name = "/scratch/izar/cizinsky/pretrained/huggingface/models--3DAIGC--LHM-1B/snapshots/cd8a1cc900a557d83187cfc2e0a91cef3eba969d/"
        return hf_model_cls.from_pretrained(model_name)

    def _apply_loaded_skinning(self):
        if self.loaded_skinning is None:
            return
        if not hasattr(self.model.renderer, "smplx_model"):
            return
        smplx = self.model.renderer.smplx_model
        if "voxel_ws" in self.loaded_skinning and hasattr(smplx, "voxel_ws"):
            smplx.voxel_ws.data.copy_(self.loaded_skinning["voxel_ws"].to(self.tuner_device))
        if "skinning_weight" in self.loaded_skinning and hasattr(smplx, "skinning_weight"):
            smplx.skinning_weight.data.copy_(self.loaded_skinning["skinning_weight"].to(self.tuner_device))

    def _load_gs_model(self, root_output_dir: Path):
        refined_dir = root_output_dir / "refined_scene_recon" / self.cfg.exp_name
        root_gs_model_dir = refined_dir if (refined_dir / "00").exists() else root_output_dir / "initial_scene_recon"
        track_ids = sorted(
            [
                track_id
                for track_id in os.listdir(root_gs_model_dir)
                if (root_gs_model_dir / track_id).is_dir()
            ]
        )
        # print(f"[DEBUG] Found {len(track_ids)} humans for finetuning: {track_ids}")
        self.all_model_list = []
        self.track_meta = []
        for track_id in track_ids:
            # print(f"    Loading GS model for track_id: {track_id}")
            gs_model_dir = root_gs_model_dir / track_id
            gs_model_list = torch.load(gs_model_dir / "gs_model_list.pt", map_location=self.tuner_device)
            query_points = torch.load(gs_model_dir / "query_points.pt", map_location=self.tuner_device)
            transform_mat_neutral_pose = torch.load(
                gs_model_dir / "transform_mat_neutral_pose.pt", map_location=self.tuner_device
            )
            motion_seq = torch.load(gs_model_dir / "motion_seq.pt", map_location=self.tuner_device)
            # print(f"       Motion_seq root_pose shape: {motion_seq['smplx_params']['root_pose'].shape}")
            shape_params = torch.from_numpy(np.load(gs_model_dir / "shape_params.npy")).unsqueeze(0).to(self.tuner_device)
            model = (gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_params)
            self.all_model_list.append(model)
            motion_seq_for_save = copy.deepcopy(motion_seq)
            self.track_meta.append(
                {
                    "track_id": track_id,
                    "gs_count": len(gs_model_list),
                    "query_count": query_points.shape[0],
                    "transform_count": transform_mat_neutral_pose.shape[0],
                    "motion_seq": motion_seq_for_save,
                    "shape_params": shape_params,
                }
            )
        # print(f"[DEBUG] Loaded GS models for {len(self.all_model_list)} humans.")
        # Load trainable skinning weights if present (shared across tracks)
        skin_path = root_gs_model_dir.parent if root_gs_model_dir.name == self.cfg.exp_name else root_output_dir
        skin_file = root_output_dir / "smplx_skinning.pt"
        if refined_dir.exists():
            skin_file = refined_dir / "smplx_skinning.pt"
        if skin_file.exists():
            self.loaded_skinning = torch.load(skin_file, map_location=self.tuner_device)

    def _prepare_joined_inputs(self):
        train_fields = tuple(self.train_params)
        self.gs_model_list: List[GaussianAppOutput] = []
        self.query_points = None
        self.transform_mat_neutral_pose = None
        self.motion_seq = None
        self.shape_param = None

        self.gs_track_offsets = []
        self.query_track_offsets = []

        gs_cursor = 0
        query_cursor = 0

        for track_idx, packed in enumerate(self.all_model_list):
            (
                p_gs_model_list,
                p_query_points,
                p_transform_mat_neutral_pose,
                p_motion_seq,
                p_shape_param,
            ) = packed

            # Make specified fields trainable.
            enable_gaussian_grads(p_gs_model_list[0], train_fields, detach_to_leaf=True)

            self.gs_track_offsets.append((gs_cursor, len(p_gs_model_list)))
            self.query_track_offsets.append((query_cursor, p_query_points.shape[0], p_transform_mat_neutral_pose.shape[0]))
            gs_cursor += len(p_gs_model_list)
            query_cursor += p_query_points.shape[0]

            self.gs_model_list.extend(p_gs_model_list)

            if self.query_points is None:
                self.query_points = p_query_points
            else:
                self.query_points = torch.cat([self.query_points, p_query_points], dim=0)

            if self.transform_mat_neutral_pose is None:
                self.transform_mat_neutral_pose = p_transform_mat_neutral_pose
            else:
                self.transform_mat_neutral_pose = torch.cat(
                    [self.transform_mat_neutral_pose, p_transform_mat_neutral_pose], dim=0
                )

            if self.motion_seq is None:
                self.motion_seq = copy.deepcopy(p_motion_seq)
            else:
                for key in self.motion_seq["smplx_params"].keys():
                    self.motion_seq["smplx_params"][key] = torch.cat(
                        [self.motion_seq["smplx_params"][key], p_motion_seq["smplx_params"][key]],
                        dim=0,
                    )

            if self.shape_param is None:
                self.shape_param = p_shape_param
            else:
                self.shape_param = torch.cat([self.shape_param, p_shape_param], dim=0)

        print(f"[DEBUG] len of gs model list: {len(self.gs_model_list)}")
        print(f"[DEBUG] shape of query points: {self.query_points.shape}")
        print(f"[DEBUG] shape of transform_mat_neutral_pose: {self.transform_mat_neutral_pose.shape}")
        print(f"[DEBUG] shape of shape param: {self.shape_param.shape}")
        for k, v in self.motion_seq["smplx_params"].items():
            print(f"[DEBUG] motion_seq smplx_params key:{k}, shape:{v.shape}")

    # ---------------- Training utilities ----------------
    def _trainable_tensors(self) -> List[torch.Tensor]:
        params = []
        for gauss in self.gs_model_list:
            for name in self.train_params:
                t = getattr(gauss, name, None)
                if torch.is_tensor(t) and t.requires_grad:
                    params.append(t)

        if self.cfg.tune_motion:
            # enable grads on SMPL-X params inside the renderer if present
            if hasattr(self.model.renderer, "smplx_model"):
                smplx = self.model.renderer.smplx_model
                # enable trainable skinning if available
                if hasattr(smplx, "use_trainable_skinning"):
                    print(f"[INFO] Enabling trainable skinning")
                    smplx.use_trainable_skinning = True
                for attr in ("shape_params", "pose_params", "expr_dirs", "pose_dirs", "shape_dirs", "voxel_ws", "skinning_weight"):
                    if hasattr(smplx, attr):
                        v = getattr(smplx, attr)
                        if torch.is_tensor(v):
                            v.requires_grad_(True)
                            params.append(v)
        else:
            if hasattr(self.model.renderer, "smplx_model"):
                smplx = self.model.renderer.smplx_model
                if hasattr(smplx, "use_trainable_skinning"):
                    print(f"[INFO] Disabling trainable skinning")
                    smplx.use_trainable_skinning = False
                for attr in ("voxel_ws", "skinning_weight"):
                    if hasattr(smplx, attr):
                        v = getattr(smplx, attr)
                        if torch.is_tensor(v):
                            v.requires_grad_(False)
        return params

    def _slice_motion(self, frame_indices: torch.Tensor):
        keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
            "focal",
            "princpt",
            "img_size_wh",
            "expr",
        ]
        smplx = {"betas": self.shape_param.to(self.tuner_device)}
        smplx["transform_mat_neutral_pose"] = self.transform_mat_neutral_pose
        for key in keys:
            smplx[key] = torch.index_select(
                self.motion_seq["smplx_params"][key], 1, frame_indices.to(self.motion_seq["smplx_params"][key].device)
            ).to(self.tuner_device)

        render_c2ws = torch.index_select(self.motion_seq["render_c2ws"], 1, frame_indices).to(self.tuner_device)
        render_intrs = torch.index_select(self.motion_seq["render_intrs"], 1, frame_indices).to(self.tuner_device)
        render_bg_colors = torch.index_select(self.motion_seq["render_bg_colors"], 1, frame_indices).to(self.tuner_device)
        return smplx, render_c2ws, render_intrs, render_bg_colors

    def _render_batch(self, frame_indices: torch.Tensor):
        smplx_params, render_c2ws, render_intrs, render_bg_colors = self._slice_motion(frame_indices)
        # Override background to black
        render_bg_colors = torch.zeros_like(render_bg_colors)
        return self.model.animation_infer_custom(
            self.gs_model_list,
            self.query_points,
            smplx_params,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
        )

    def _canonical_regularization(self):
        """Return combined canonical regularization and its components."""
        asap_terms = []
        acap_terms = []
        margin = 0.0525  # meters
        for gauss in self.gs_model_list:
            # Gaussian Shape Regularization (ASAP): encourage isotropic scales.
            scales = gauss.scaling
            # If scaling is stored in log-space, exp() keeps positivity; otherwise it’s a smooth surrogate.
            scales_pos = torch.exp(scales)
            asap = ((scales_pos - scales_pos.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean()
            asap_terms.append(asap)

            # Positional anchoring (ACAP): hinge on offset magnitude beyond margin.
            offsets = gauss.offset_xyz
            acap = torch.clamp(offsets.norm(dim=-1) - margin, min=0.0).mean()
            acap_terms.append(acap)

        if len(asap_terms) == 0:
            return torch.tensor(0.0, device=self.tuner_device), torch.tensor(0.0, device=self.tuner_device), torch.tensor(
                0.0, device=self.tuner_device
            )

        asap_loss = torch.stack(asap_terms).mean() * self.cfg.loss_weights["reg_asap"]
        acap_loss = torch.stack(acap_terms).mean() * self.cfg.loss_weights["reg_acap"]

        return asap_loss, acap_loss

    def _compute_depth_loss(self, pred_depth, masks, frame_paths, render_hw):
        """Compute masked L1 depth loss and return (loss, gt_depth, valid_mask)."""
        zero = torch.tensor(0.0, device=self.tuner_device)
        if pred_depth is None or self.cfg.loss_weights.get("depth", 0) <= 0:
            return zero, None, None

        gt_depths = []
        valid_masks = []
        for i, fp in enumerate(frame_paths):
            depth_path = self.depth_dir / (Path(fp).stem + ".npy")
            if depth_path.exists():
                depth_np = np.load(depth_path)
                depth_t = torch.from_numpy(depth_np).float().to(self.tuner_device)
            else:
                depth_t = torch.zeros(render_hw, device=self.tuner_device)
            depth_t = depth_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            depth_t = F.interpolate(depth_t, size=render_hw, mode="bilinear", align_corners=False)
            gt_depths.append(depth_t)

            m = masks[i : i + 1, ..., 0:1].permute(0, 3, 1, 2)  # [1,1,H,W]
            m = F.interpolate(m, size=render_hw, mode="nearest")
            valid = torch.isfinite(depth_t) & (depth_t > 0) & (m > 0.5)
            valid_masks.append(valid.float())

        gt_depth = torch.cat(gt_depths, dim=0)  # [B,1,H,W]
        valid_mask = torch.cat(valid_masks, dim=0)
        pred_d = pred_depth.permute(0, 3, 1, 2)  # [B,1,H,W]
        diff = torch.abs(pred_d - gt_depth) * valid_mask
        denom = valid_mask.sum().clamp_min(1e-6)
        depth_loss = diff.sum() / denom
        return depth_loss, gt_depth, valid_mask

    # ---------------- Logging utilities ----------------
    def _init_wandb(self):
        if not self.cfg.wandb.enable or wandb is None:
            return
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                    "grad_clip": self.cfg.grad_clip,
                    "train_params": self.train_params,
                    "exp_name": self.cfg.exp_name,
                    "scene_name": self.cfg.scene_name,
                    "output_dir": str(self.output_dir),
                    "loss_weights": self.cfg.loss_weights,
                    "sample_every": self.cfg.sample_every,
                },
                name=self.cfg.exp_name,
                tags=list(self.cfg.wandb.tags) if "tags" in self.cfg.wandb else None,
            )

    # ---------------- Training loop ----------------
    def train_loop(self):
        if self.wandb_run is None:
            self._init_wandb()

        dataset = FrameMaskDataset(
            self.frames_dir, self.masks_dir, self.tuner_device, sample_every=self.cfg.sample_every
        )
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, drop_last=False
        )

        params = self._trainable_tensors()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Pre-optimization visualization (epoch 0).
        if self.cfg.vis_every_epoch > 0:
            for tidx in range(len(self.track_meta)):
                self._canonical_vis_for_track(tidx, 0)
        if getattr(self.cfg, "eval_every_epoch", 0) > 0:
            self.eval_loop(0)

        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            batch = 0
            for frame_indices, frames, masks, frame_paths in pbar:
                optimizer.zero_grad(set_to_none=True)

                frame_indices = frame_indices.to(self.tuner_device)
                res = self._render_batch(frame_indices)
                comp_rgb = res["comp_rgb"]  # [B, H, W, 3], 0-1
                comp_mask = res["comp_mask"][..., :1]  # [B, H, W, 1]
                pred_depth = res.get("comp_depth", None)  # [B, H, W, 1] if available

                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3
                # Use raw comp_rgb against masked GT; keep weights outside the loss helpers.
                rgb_loss = F.mse_loss(comp_rgb, gt_masked)
                sil_loss = F.mse_loss(comp_mask, masks)
                # SSIM (unmasked) between comp_rgb and masked GT
                ssim_val = fused_ssim(_ensure_nchw(comp_rgb), _ensure_nchw(gt_masked), padding="valid")
                ssim_loss = 1.0 - ssim_val
                depth_loss, gt_depth, valid_mask = self._compute_depth_loss(
                    pred_depth, masks, frame_paths, comp_rgb.shape[1:3]
                )
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = (
                    self.cfg.loss_weights["rgb"] * rgb_loss
                    + self.cfg.loss_weights["sil"] * sil_loss
                    + self.cfg.loss_weights["ssim"] * ssim_loss
                    + self.cfg.loss_weights.get("depth", 0) * depth_loss
                    + reg_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    asap=f"{asap_loss.item():.4f}",
                    acap=f"{acap_loss.item():.4f}",
                    depth=f"{depth_loss.item():.4f}",
                )

                if self.wandb_run is not None:
                    wandb.log(
                        {
                            "loss/combined": loss.item(),
                            "loss/rgb": rgb_loss.item(),
                            "loss/sil": sil_loss.item(),
                            "loss/ssim": ssim_loss.item(),
                            "loss/depth": depth_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                        }
                    )

                if batch in [0,2,4]:
                    # define debug save dir
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name
                    debug_save_dir.mkdir(parents=True, exist_ok=True)

                    # - create a joined image from pred_masked and gt_masked for debugging
                    joined_image = torch.cat([comp_rgb, gt_masked], dim=3)  # Concatenate along width
                    debug_image_path = debug_save_dir / f"rgb_loss_input.png"
                    save_image(joined_image.permute(0, 3, 1, 2), str(debug_image_path))

                    # - create comp mask and masks image 
                    comp_mask_image = torch.cat([comp_mask.repeat(1, 1, 1, 3), masks.repeat(1, 1, 1, 3)], dim=3)  # Concatenate along width
                    debug_comp_mask_path = debug_save_dir / f"sil_loss_input.png"
                    save_image(comp_mask_image.permute(0, 3, 1, 2), str(debug_comp_mask_path))

                    # - create ssim input image
                    ssim_input_image = torch.cat([comp_rgb, gt_masked], dim=3)  # Concatenate along width
                    debug_ssim_path = debug_save_dir / f"ssim_loss_input.png"
                    save_image(ssim_input_image.permute(0, 3, 1, 2), str(debug_ssim_path))

                    if pred_depth is not None:
                        # Depth debug for first sample in batch
                        pd = pred_depth[0, ..., 0].detach().cpu().numpy()
                        gd = gt_depth[0, 0].detach().cpu().numpy() if gt_depth is not None else None
                        vm = valid_mask[0, 0].detach().cpu().numpy() if valid_mask is not None else None

                        # Shared normalization across pred and gt
                        if gd is not None:
                            masked_vals = []
                            if vm is not None:
                                masked_vals.append(pd[vm > 0.5])
                                masked_vals.append(gd[vm > 0.5])
                            else:
                                masked_vals.append(pd)
                                masked_vals.append(gd)
                            vals = np.concatenate([v[np.isfinite(v)] for v in masked_vals if v.size > 0])
                            if vals.size > 0:
                                vmin, vmax = vals.min(), vals.max()
                            else:
                                vmin, vmax = 0.0, 1.0
                        else:
                            vmin = vmax = None

                        pred_color = depth_to_color(pd, vm, vmin, vmax)
                        if gd is not None:
                            gt_color = depth_to_color(gd, vm, vmin, vmax)
                            # Concatenate pred | gt for a single debug view
                            depth_concat = np.concatenate([pred_color, gt_color], axis=1)
                            Image.fromarray(depth_concat).save(debug_save_dir / "depth_debug.png")
                        else:
                            Image.fromarray(pred_color).save(debug_save_dir / "depth_debug.png")

                batch += 1


            avg_loss = running_loss / max(1, len(loader))
            print(f"[Epoch {epoch+1}/{self.cfg.epochs}] loss={avg_loss:.4f}")
            if self.wandb_run is not None:
                wandb.log({"loss/combined_epoch": avg_loss, "epoch": epoch + 1})

            if self.cfg.vis_every_epoch > 0 and (epoch + 1) % self.cfg.vis_every_epoch == 0:
                for tidx in range(len(self.track_meta)):
                    self._canonical_vis_for_track(tidx, epoch + 1)
            if getattr(self.cfg, "eval_every_epoch", 0) > 0 and (epoch + 1) % self.cfg.eval_every_epoch == 0:
                self.eval_loop(epoch + 1)

        self._save_refined_models()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ---------------- Evaluation -------------------
    def eval_loop(self, epoch):
        # Parse the evaluation setup
        source_camera_id: int = self.cfg.nvs_eval.source_camera_id
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)
        camera_params_path: Path = root_gt_dir_path / "cameras" / "rgb_cameras.npz"
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)

        def _extr_to_w2c_4x4(extr: torch.Tensor) -> torch.Tensor:
            w2c = torch.eye(4, device=self.tuner_device, dtype=torch.float32)
            w2c[:3, :4] = extr.to(self.tuner_device)
            return w2c

        def _intr_to_4x4(intr: torch.Tensor) -> torch.Tensor:
            intr4 = torch.eye(4, device=self.tuner_device, dtype=torch.float32)
            intr4[:3, :3] = intr.to(self.tuner_device)
            return intr4

        def _rescale_intrinsics(intr: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
            """Rescale fx/fy/cx/cy if the render resolution differs from the intrinsic's implied size."""
            intr = intr.clone()
            orig_w = intr[0, 2] * 2.0
            orig_h = intr[1, 2] * 2.0
            if orig_w <= 0 or orig_h <= 0:
                return intr
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            intr[0, 0] = intr[0, 0] * scale_x
            intr[1, 1] = intr[1, 1] * scale_y
            intr[0, 2] = intr[0, 2] * scale_x
            intr[1, 2] = intr[1, 2] * scale_y
            return intr

        # Load source camera (used to convert target cameras into source/world coords)
        src_intr, src_extr = load_camera_from_npz(camera_params_path, source_camera_id, device=self.tuner_device)
        src_w2c = _extr_to_w2c_4x4(src_extr)
        src_c2w = torch.inverse(src_w2c)

        for tgt_cam_id in target_camera_ids:
            tgt_gt_frames_dir_path = root_gt_dir_path / "images" / f"{tgt_cam_id}"
            tgt_gt_masks_dir_path = root_gt_dir_path / "seg" / "img_seg_mask" / f"{tgt_cam_id}" / "all"
            tgt_intr, tgt_extr = load_camera_from_npz(camera_params_path, tgt_cam_id, device=self.tuner_device)
            # Match render resolution to GT frames by rescaling intrinsics to the actual image size.
            if tgt_gt_frames_dir_path.exists():
                first_frame = next(iter(sorted(tgt_gt_frames_dir_path.glob("*.png")) or sorted(tgt_gt_frames_dir_path.glob("*.jpg")) or sorted(tgt_gt_frames_dir_path.glob("*.jpeg"))), None)
                if first_frame is not None:
                    with Image.open(first_frame) as img:
                        w, h = img.size
                    tgt_intr = _rescale_intrinsics(tgt_intr, w, h)
            tgt_w2c = _extr_to_w2c_4x4(tgt_extr)
            tgt_c2w_world = torch.inverse(tgt_w2c)

            # Express target camera in source-camera coordinates: S <- W <- T
            tgt_c2w_in_src = src_w2c @ tgt_c2w_world
            tgt_intr4 = _intr_to_4x4(tgt_intr)

            save_dir = root_save_dir / f"{tgt_cam_id}"
            save_dir.mkdir(parents=True, exist_ok=True)

            dataset = FrameMaskDataset(tgt_gt_frames_dir_path, tgt_gt_masks_dir_path, self.tuner_device, sample_every=1)
            loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

            # Use target camera pose expressed in source frame.
            tgt_c2w_template = tgt_c2w_in_src.unsqueeze(0).unsqueeze(0)  # [1,1,4,4]
            tgt_intr_template = tgt_intr4.unsqueeze(0).unsqueeze(0)      # [1,1,4,4]

            with torch.no_grad():
                for frame_indices, frames, masks, frame_paths in tqdm(loader, desc=f"NVS cam {tgt_cam_id}", leave=False):
                    frame_indices = frame_indices.to(self.tuner_device)
                    smplx_params, _, _, render_bg_colors = self._slice_motion(frame_indices)

                    # Build camera tensors matching render_bg_colors leading dims [B?, F]
                    num_tracks, num_frames = render_bg_colors.shape[0], render_bg_colors.shape[1]
                    render_c2ws = tgt_c2w_template.expand(num_tracks, num_frames, 4, 4)
                    render_intrs = tgt_intr_template.expand(num_tracks, num_frames, 4, 4)
                    render_bg_colors = torch.zeros((num_tracks, num_frames, 3), device=self.tuner_device, dtype=torch.float32)

                    res = self.model.animation_infer_custom(
                        self.gs_model_list,
                        self.query_points,
                        smplx_params,
                        render_c2ws=render_c2ws,
                        render_intrs=render_intrs,
                        render_bg_colors=render_bg_colors,
                    )

                    comp_rgb = res["comp_rgb"]  # [B, H, W, 3]
                    # If render size differs from GT, resize render to match
                    if comp_rgb.shape[1:3] != frames.shape[1:3]:
                        comp_rgb = F.interpolate(
                            comp_rgb.permute(0, 3, 1, 2),
                            size=frames.shape[1:3],
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                    masks3 = masks
                    if masks3.shape[-1] == 1:
                        masks3 = masks3.repeat(1, 1, 1, 3)
                    masked_render = comp_rgb # * masks3
                    masked_gt = frames * masks3
                    joined = torch.cat([masked_render, masked_gt], dim=2)  # side-by-side along width

                    for bi in range(frames.shape[0]):
                        fname = Path(frame_paths[bi]).name
                        save_image(joined[bi], str(save_dir / fname))




    # ---------------- Visualization ----------------
    def _canonical_vis_for_track(self, track_idx: int, epoch: int):
        gs_start, gs_count = self.gs_track_offsets[track_idx]
        q_start, q_count, t_count = self.query_track_offsets[track_idx]

        gs_slice = self.gs_model_list[gs_start : gs_start + gs_count]
        query_slice = self.query_points[q_start : q_start + q_count]
        transform_slice = self.transform_mat_neutral_pose[q_start : q_start + t_count]
        meta = self.track_meta[track_idx]
        motion_seq = meta["motion_seq"]
        shape_params = meta["shape_params"]

        centers = compute_frame_centers_from_smplx(motion_seq["smplx_params"]).to(self.tuner_device)
        num_frames = motion_seq["render_c2ws"].shape[1]

        view_degs = [0, 90, 180, 270]
        for deg in view_degs:
            view_dir = (
                self.output_dir
                / "refined_scene_recon"
                / self.cfg.exp_name
                / "visualisations"
                / meta["track_id"]
                / f"view_deg{deg}"
            )
            view_dir.mkdir(parents=True, exist_ok=True)
            video_path = view_dir / f"epoch_{epoch:04d}.mp4"
            if epoch == 0 and video_path.exists():
                continue

            frames = []
            for fi in tqdm(range(num_frames), desc=f"Vis track {meta['track_id']} view {deg} epoch {epoch}"):
                smplx_params = {
                    "betas": shape_params.to(self.tuner_device),
                    "transform_mat_neutral_pose": transform_slice.to(self.tuner_device),
                }
                for k, v in motion_seq["smplx_params"].items():
                    if k == "betas":
                        smplx_params[k] = shape_params.to(self.tuner_device)
                    else:
                        smplx_params[k] = v[:, fi : fi + 1].to(self.tuner_device)

                if deg == 0:
                    render_c2ws = motion_seq["render_c2ws"][:, fi : fi + 1].to(self.tuner_device)
                else:
                    rotated = rotate_c2ws_y_about_center(
                        motion_seq["render_c2ws"], centers, degrees=deg
                    ).to(self.tuner_device)
                    render_c2ws = rotated[:, fi : fi + 1]

                render_intrs = motion_seq["render_intrs"][:, fi : fi + 1].to(self.tuner_device)
                render_bg_colors = motion_seq["render_bg_colors"][:, fi : fi + 1].to(self.tuner_device)

                with torch.no_grad():
                    res = self.model.animation_infer_custom(
                        gs_slice,
                        query_slice,
                        smplx_params,
                        render_c2ws=render_c2ws,
                        render_intrs=render_intrs,
                        render_bg_colors=render_bg_colors,
                    )
                    img = res["comp_rgb"][0].detach().cpu().clamp(0, 1).numpy()

                img_uint8 = (img * 255).astype(np.uint8)
                frames.append(img_uint8)

            try:
                writer = imageio.get_writer(video_path, fps=10, codec="libx264", format="ffmpeg")
                for f in frames:
                    writer.append_data(f)
                writer.close()
                shutil.copy(video_path, view_dir / "last.mp4")
            except Exception as e:
                print(f"[WARN] video export failed for view {deg}: {e}")

    # ---------------- Saving ----------------
    def _save_refined_models(self):
        save_root = self.output_dir / "refined_scene_recon" / self.cfg.exp_name
        save_root.mkdir(parents=True, exist_ok=True)

        # print(f"[INFO] Saving refined models to {save_root}...")
        for idx, meta in enumerate(self.track_meta):
            track_id = meta["track_id"]
            # print(f"    Saving refined model for track_id: {track_id}")
            gs_start, gs_count = self.gs_track_offsets[idx]
            q_start, q_count, t_count = self.query_track_offsets[idx]

            track_dir = save_root / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            gs_slice = self.gs_model_list[gs_start : gs_start + gs_count]
            query_slice = self.query_points[q_start : q_start + q_count].detach().cpu()
            transform_slice = self.transform_mat_neutral_pose[q_start : q_start + t_count].detach().cpu()

            torch.save(gs_slice, track_dir / "gs_model_list.pt")
            torch.save(query_slice, track_dir / "query_points.pt")
            torch.save(transform_slice, track_dir / "transform_mat_neutral_pose.pt")
            torch.save(meta["motion_seq"], track_dir / "motion_seq.pt")
            np.save(track_dir / "shape_params.npy", meta["shape_params"].squeeze(0).cpu().numpy())

        print(f"[INFO] Saved refined models to {save_root}")
        # Save shared skinning weights if available
        if hasattr(self.model.renderer, "smplx_model"):
            smplx = self.model.renderer.smplx_model
            payload = {}
            if hasattr(smplx, "voxel_ws"):
                payload["voxel_ws"] = smplx.voxel_ws.detach().cpu()
            if hasattr(smplx, "skinning_weight"):
                payload["skinning_weight"] = smplx.skinning_weight.detach().cpu()
            if payload:
                torch.save(payload, save_root / "smplx_skinning.pt")

    def infer(self):
        # keep this here for compatibility
        pass

    def infer_single(self):
        # keep this here for compatibility
        pass


@hydra.main(config_path="configs", config_name="finetune", version_base="1.3")
def main(cfg: DictConfig):
    tuner = MultiHumanFinetuner(cfg)
    tuner.train_loop()


if __name__ == "__main__":
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
