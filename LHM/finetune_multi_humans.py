"""
Finetune canonical Gaussian params for multiple humans using original frames as supervision.

Trains offset_xyz, rotation, scaling, opacity, and SH/RGB while keeping SMPL-X motion fixed.
Supervision:
  - L_rgb: MSE between rendered RGB (masked) and GT frame * union mask (weight 1.0)
  - L_sil: MSE between rendered mask and GT union mask (weight 0.5)
  - L_lpips: masked LPIPS (weight 1.0)

Expects:
  output_dir/initial_scene_recon/<track_id>/* (loaded canonical state)
  output_dir/frames/<0000.png...> and output_dir/masks/union/<0000.png...> for supervision.
Saves finetuned state to output_dir/refined_scene_recon/<track_id>/...
"""


import os
import sys
import shutil
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


# ---------------------------------------------------------------------------
# Loss helpers (LPIPS masked)
# ---------------------------------------------------------------------------
_LPIPS_METRIC = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
        for p in _LPIPS_METRIC.parameters():
            p.requires_grad = False
    return _LPIPS_METRIC.to(device)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2).contiguous()


def masked_lpips(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """LPIPS averaged over masked region per sample."""
    target = _ensure_nchw(images.clamp(0.0, 1.0))
    preds = _ensure_nchw(renders.clamp(0.0, 1.0))
    if masks.dim() == 4 and masks.shape[-1] == 1:
        mask = masks[..., 0]
    else:
        mask = masks
    mask = mask.unsqueeze(1).float()  # [B,1,H,W]

    net = _get_lpips_net(preds.device)
    dmap = net(preds, target)

    if dmap.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask, size=dmap.shape[-2:], mode="nearest")
    else:
        mask_resized = mask

    numerator = (dmap * mask_resized).sum(dim=(1, 2, 3))
    denom = mask_resized.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    vals = numerator / denom
    vals = torch.where(denom < 1e-5, torch.zeros_like(vals), vals)
    return vals.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FrameMaskDataset(Dataset):
    def __init__(self, frames_dir: Path, masks_dir: Path, device: torch.device, sample_every: int = 1):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.device = device

        self.frame_paths = sorted(frames_dir.glob("*.png"))
        if sample_every > 1:
            self.frame_paths = self.frame_paths[::sample_every]
        if not self.frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}")
        self.mask_paths = [masks_dir / p.name for p in self.frame_paths]
        missing = [p for p in self.mask_paths if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing masks for frames: {missing[:5]}")

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
        return torch.tensor(idx, device=self.device, dtype=torch.long), frame, mask


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

        self._load_gs_model(self.output_dir)
        self._prepare_joined_inputs()
        self.model: ModelHumanLRMSapdinoBodyHeadSD3_5 = self._build_model().to(self.tuner_device)

        self.frames_dir = self.output_dir / "frames"
        self.masks_dir = self.output_dir / "masks" / "union"

#        # clean this dir
        #save_root = self.output_dir / "refined_scene_recon" / self.cfg.exp_name
        #if save_root.exists():
            #shutil.rmtree(save_root)

    # ---------------- Model / data loading ----------------
    def _build_model(self):
        hf_model_cls = wrap_model_hub(ModelHumanLRMSapdinoBodyHeadSD3_5)
        model_name = "/scratch/izar/cizinsky/pretrained/huggingface/models--3DAIGC--LHM-1B/snapshots/cd8a1cc900a557d83187cfc2e0a91cef3eba969d/"
        return hf_model_cls.from_pretrained(model_name)

    def _load_gs_model(self, root_output_dir: Path):
        root_gs_model_dir = root_output_dir / "initial_scene_recon"
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

        if self.cfg.tune_smplx:
            # enable grads on SMPL-X params inside the renderer if present
            if hasattr(self.model.renderer, "smplx_model"):
                smplx = self.model.renderer.smplx_model
                for attr in ("shape_params", "pose_params", "expr_dirs", "pose_dirs", "shape_dirs"):
                    if hasattr(smplx, attr):
                        v = getattr(smplx, attr)
                        if torch.is_tensor(v):
                            v.requires_grad_(True)
                            params.append(v)
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

        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            batch = 0
            for frame_indices, frames, masks in pbar:
                optimizer.zero_grad(set_to_none=True)

                frame_indices = frame_indices.to(self.tuner_device)
                res = self._render_batch(frame_indices)
                comp_rgb = res["comp_rgb"]  # [B, H, W, 3], 0-1
                comp_mask = res["comp_mask"][..., :1]  # [B, H, W, 1]

                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3
                pred_masked = comp_rgb # * mask3

                rgb_loss = F.mse_loss(pred_masked, gt_masked) *self.cfg.loss_weights["rgb"]
                sil_loss = F.mse_loss(comp_mask, masks) * self.cfg.loss_weights["sil"]
                lpips_loss = masked_lpips(gt_masked, masks, comp_rgb) * self.cfg.loss_weights["lpips"]
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = rgb_loss + sil_loss + lpips_loss + reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    asap=f"{asap_loss.item():.4f}",
                    acap=f"{acap_loss.item():.4f}",
                )

                if self.wandb_run is not None:
                    wandb.log(
                        {
                            "loss/combined": loss.item(),
                            "loss/rgb": rgb_loss.item(),
                            "loss/sil": sil_loss.item(),
                            "loss/lpips": lpips_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                        }
                    )

                if batch in [1, 4]:
                    # define debug save dir
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name
                    debug_save_dir.mkdir(parents=True, exist_ok=True)

                    # - create a joined image from pred_masked and gt_masked for debugging
                    joined_image = torch.cat([pred_masked, gt_masked], dim=3)  # Concatenate along width
                    debug_image_path = debug_save_dir / f"rgb_loss_input.png"
                    save_image(joined_image.permute(0, 3, 1, 2), str(debug_image_path))

                    # - create comp mask and masks image 
                    comp_mask_image = torch.cat([comp_mask.repeat(1, 1, 1, 3), masks.repeat(1, 1, 1, 3)], dim=3)  # Concatenate along width
                    debug_comp_mask_path = debug_save_dir / f"sil_loss_input.png"
                    save_image(comp_mask_image.permute(0, 3, 1, 2), str(debug_comp_mask_path))

                    # - create lpips input image
                    lpips_input_image = torch.cat([comp_rgb, gt_masked], dim=3)  # Concatenate along width
                    debug_lpips_path = debug_save_dir / f"lpips_loss_input.png"
                    save_image(lpips_input_image.permute(0, 3, 1, 2), str(debug_lpips_path))

                batch += 1


            avg_loss = running_loss / max(1, len(loader))
            print(f"[Epoch {epoch+1}/{self.cfg.epochs}] loss={avg_loss:.4f}")
            if self.wandb_run is not None:
                wandb.log({"loss/combined_epoch": avg_loss, "epoch": epoch + 1})

            if self.cfg.vis_every_epoch > 0 and (epoch + 1) % self.cfg.vis_every_epoch == 0:
                for tidx in range(len(self.track_meta)):
                    self._canonical_vis_for_track(tidx, epoch + 1)

        self._save_refined_models()
        if self.wandb_run is not None:
            self.wandb_run.finish()

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

        # Use first-frame SMPL-X params (avoid zeroing to keep subject in view).
        smplx_params = {
            "betas": shape_params.to(self.tuner_device),  # [1,10]
            "transform_mat_neutral_pose": transform_slice.to(self.tuner_device),
        }
        for k, v in motion_seq["smplx_params"].items():
            if k == "betas":
                smplx_params[k] = shape_params.to(self.tuner_device)
            else:
                smplx_params[k] = v[:, 0:1].to(self.tuner_device)

        centers = compute_frame_centers_from_smplx(motion_seq["smplx_params"]).to(self.tuner_device)

        view_degs = [0, 90, 180, 270]
        for deg in view_degs:
            if deg == 0:
                render_c2ws = motion_seq["render_c2ws"][:, 0:1].to(self.tuner_device)
            else:
                rotated = rotate_c2ws_y_about_center(
                    motion_seq["render_c2ws"], centers, degrees=deg
                ).to(self.tuner_device)
                render_c2ws = rotated[:, 0:1]

            render_intrs = motion_seq["render_intrs"][:, 0:1].to(self.tuner_device)
            render_bg_colors = motion_seq["render_bg_colors"][:, 0:1].to(self.tuner_device)

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
            save_dir = (
                self.output_dir
                / "refined_scene_recon"
                / self.cfg.exp_name
                / "visualisations"
                / meta["track_id"]
                / f"view_deg{deg}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            epoch_name = f"{epoch:04d}.png"
            epoch_path = save_dir / epoch_name
            Image.fromarray(img_uint8).save(epoch_path)
            # Copy to last.png
            Image.fromarray(img_uint8).save(save_dir / "last.png")

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
