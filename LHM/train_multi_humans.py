import os
import sys
import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from typing import List, Tuple, Optional
import copy
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import kornia
from torch.utils.data import DataLoader, Dataset

from fused_ssim import fused_ssim
import pyiqa
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LHM.outputs.output import GaussianAppOutput
from LHM.models.rendering.custom_gs_renderer import GS3DRenderer
from LHM.debug import overlay_smplx_mesh_pyrender

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

# Cached LPIPS metric instance; built lazily on first use.
_LPIPS_METRIC: Optional[torch.nn.Module] = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        # Spatial LPIPS gives a per-pixel distance map (pyiqa handles input normalisation to [-1,1])
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
    return _LPIPS_METRIC.to(device)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Convert tensors from NHWC (renderer output) to NCHW for library calls."""
    return t.permute(0, 3, 1, 2).contiguous()


def _mask_sums(mask: torch.Tensor) -> torch.Tensor:
    """Sum mask activations per sample (expects shape [B,1,H,W])."""
    return mask.sum(dim=(2, 3))


def ssim(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked SSIM per sample.
    
    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns: 
        ssim_vals: [B] masked SSIM values 
    """
    target = _ensure_nchw(images.float())
    preds = _ensure_nchw(renders.float())
    mask = masks.unsqueeze(1).float()

    # Kornia returns per-channel SSIM; average across channels before masking
    ssim_map = kornia.metrics.ssim(preds, target, window_size=11, max_val=1.0)
    ssim_map = ssim_map.mean(1, keepdim=True)

    # Reduce over the masked region for each batch element
    numerator = (ssim_map * mask).sum(dim=(2, 3))
    mask_sum = _mask_sums(mask)
    safe_mask_sum = mask_sum.clamp_min(1e-6)
    result = numerator / safe_mask_sum
    result = torch.where(mask_sum < 1e-5, torch.zeros_like(result), result)
    return result.squeeze(1)


def psnr(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Masked PSNR per sample.

    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]

    Returns:
        psnr_vals: [B] masked PSNR values 
    """
    target = images.float()
    preds = renders.float()
    mask = masks.unsqueeze(-1).float()

    diff2 = (preds - target) ** 2
    masked_diff2 = diff2 * mask
    # Compute masked MSE then convert to PSNR
    numerator = masked_diff2.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    mse = numerator / safe_denom
    mse = mse.clamp_min(1e-12)
    psnr_vals = 10.0 * torch.log10((max_val ** 2) / mse)
    psnr_vals = torch.where(denom < 1e-5, torch.zeros_like(psnr_vals), psnr_vals)
    return psnr_vals


def lpips(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked spatial LPIPS per sample using pyiqa.


    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns:
        lpips_vals: [B] masked LPIPS values 
    """
    target = _ensure_nchw(images.float()).clamp(0.0, 1.0)
    preds = _ensure_nchw(renders.float()).clamp(0.0, 1.0)
    mask = masks.unsqueeze(1).float()

    net = _get_lpips_net(preds.device)
    with torch.no_grad():
        dmap = net(preds, target)

    # Match mask resolution to the LPIPS map and average within the mask
    if dmap.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask, size=dmap.shape[-2:], mode="nearest")
    else:
        mask_resized = mask

    numerator = (dmap * mask_resized).sum(dim=(1, 2, 3))
    denom = mask_resized.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    lpips_vals = numerator / safe_denom
    lpips_vals = torch.where(denom < 1e-5, torch.zeros_like(lpips_vals), lpips_vals)
    return lpips_vals


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

    def _load_mask(self, path: Path, eps: float = 0.05) -> torch.Tensor:
        arr = torch.from_numpy(np.array(Image.open(path))).float()  # HxWxC or HxW
        if arr.dim() == 2:
            arr = arr.unsqueeze(-1) / 255.0  # HxWx1
            return arr.to(self.device) # already binary mask

        if arr.shape[-1] == 4:
            arr = arr[..., :3] # drop alpha
        # Foreground is any pixel whose max channel exceeds eps*255
        mask = (arr.max(dim=-1).values > eps * 255).float()  # HxW
        return mask.to(self.device).unsqueeze(-1)  # HxWx1, range [0,1]

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
# Utility functions
# ---------------------------------------------------------------------------

def extr_to_w2c_4x4(extr: torch.Tensor, device) -> torch.Tensor:
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :4] = extr.to(device)
    return w2c

def intr_to_4x4(intr: torch.Tensor, device) -> torch.Tensor:
    intr4 = torch.eye(4, device=device, dtype=torch.float32)
    intr4[:3, :3] = intr.to(device)
    return intr4

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


def build_renderer():
    renderer = GS3DRenderer(
        human_model_path="/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files",
        subdivide_num=1,
        smpl_type="smplx_2",
        feat_dim=1024,
        query_dim=1024,
        use_rgb=True,
        sh_degree=3,
        mlp_network_config={'activation': 'silu', 'n_hidden_layers': 2, 'n_neurons': 512},
        xyz_offset_max_step=1.0,
        clip_scaling=[100, 0.01, 0.05, 3000],
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        fix_opacity=False,
        fix_rotation=False,
        decoder_mlp=False,
        skip_decoder=True,
        decode_with_extra_info=None,
        gradient_checkpointing=True,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,
    )

    return renderer


def get_masks_based_bbox(masks: torch.Tensor, pad: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes around masks with optional padding.

    Args:
        masks: Tensor of shape [B, H, W, 1] with mask values in [0, 1].
        pad: Number of pixels to pad the bounding box on each side.

    Returns:
        List of bounding boxes [(y_min, y_max, x_min, x_max)] for each mask in the batch.
    """

    per_item_bboxes = []
    global_bbox = None

    for b in range(masks.shape[0]):
        ys, xs = torch.where(masks[b, :, :, 0] > 0.5)
        # No mask, use full image
        if ys.numel() == 0 or xs.numel() == 0:
            y_min, y_max = 0, masks.shape[1] - 1
            x_min, x_max = 0, masks.shape[2] - 1
        else:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            # Add padding
            y_center = (y_min + y_max) / 2
            x_center = (x_min + x_max) / 2
            h_half = (y_max - y_min) / 2 + pad
            w_half = (x_max - x_min) / 2 + pad
            y_min = max(int(y_center - h_half), 0)
            y_max = min(int(y_center + h_half), masks.shape[1] - 1)
            x_min = max(int(x_center - w_half), 0)
            x_max = min(int(x_center + w_half), masks.shape[2] - 1)

        bbox = (y_min, y_max, x_min, x_max)
        per_item_bboxes.append(bbox)

        if global_bbox is None:
            global_bbox = bbox
        else:
            global_bbox = (
                min(global_bbox[0], y_min),
                max(global_bbox[1], y_max),
                min(global_bbox[2], x_min),
                max(global_bbox[3], x_max),
            )

    if global_bbox is None:
        return []

    return [global_bbox for _ in per_item_bboxes]

def bbox_crop(bboxes: List[Tuple[int, int, int, int]], to_crop: torch.Tensor) -> torch.Tensor:
    """
    Crop a batch of tensors according to the provided bounding boxes.

    Args:
        bboxes: List of bounding boxes [(y_min, y_max, x_min, x_max)] for each tensor in the batch.
        to_crop: Tensor of shape [B, H, W, C] to be cropped where C can be any number of channels.

    Returns:
        Cropped tensor of shape [B, H_crop, W_crop, C].
    """

    # Crop tensors
    cropped = []
    for b in range(to_crop.shape[0]):
        y_min, y_max, x_min, x_max = bboxes[b]
        cropped.append(to_crop[b, y_min : y_max + 1, x_min : x_max + 1, :])

    return torch.stack(cropped, dim=0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MultiHumanTrainer:
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
        self.renderer : GS3DRenderer = build_renderer().to(self.tuner_device)

        self.frames_dir = self.output_dir / "frames"
        self.masks_dir = self.output_dir / "masks" / "union"
        self.depth_dir = self.output_dir / "depth_maps" / "raw"

        self._load_gt_parameters()


    # ---------------- Model / data loading ----------------
    def _load_gt_parameters(self):

        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)

        # Load and prepare the gt camera parameters
        camera_params_path: Path = root_gt_dir_path / "cameras" / "rgb_cameras.npz"
        cam_id = self.cfg.nvs_eval.source_camera_id
        intr, extr = load_camera_from_npz(camera_params_path, cam_id, device=self.tuner_device)
        w2c = extr_to_w2c_4x4(extr, self.tuner_device)
        self.c2w = torch.inverse(w2c) # shape is [4, 4]
        self.K = intr_to_4x4(intr, self.tuner_device) # shape is [4,4]

        # SMPLX
        smplx_dir: Path = root_gt_dir_path / "smplx"
        frame_paths = sorted([p for p in os.listdir(smplx_dir) if p.endswith(".npz")])
        npzs = []
        for fp in frame_paths:
            npz = np.load(smplx_dir / f"{Path(fp).stem}.npz")
            npzs.append(npz)

        def stack_key(key):
            arrs = [torch.from_numpy(n[key]).float() for n in npzs]
            return torch.stack(arrs, dim=1).to(self.tuner_device)  # [P, F, ...]

        betas = stack_key("betas")[:, 0, :10] # [P, 10]
        gt_smplx = {
            "betas": betas,
            "root_pose": stack_key("root_pose"),   # [P,F,3] world axis-angle
            "body_pose": stack_key("body_pose"),
            "jaw_pose": stack_key("jaw_pose"),
            "leye_pose": stack_key("leye_pose"),
            "reye_pose": stack_key("reye_pose"),
            "lhand_pose": stack_key("lhand_pose"),
            "rhand_pose": stack_key("rhand_pose"),
            "trans": stack_key("trans"),           # [P,F,3] world translation
            "expr": stack_key("expression"),
        }

        gt_smplx["expr"] = torch.zeros(gt_smplx["expr"].shape[0], gt_smplx["expr"].shape[1], 100, device=self.tuner_device)

        self.gt_query_points, self.gt_smplx = self.renderer.get_query_points(gt_smplx, self.tuner_device)

        # Canonical 3DGS
        root_gs_model_dir = self.output_dir / "initial_scene_recon"
        track_ids = sorted(
            [
                track_id
                for track_id in os.listdir(root_gs_model_dir)
                if (root_gs_model_dir / track_id).is_dir()
            ]
        )
        self.gt_gs_model_list = []
        for track_id in track_ids:
            gs_model_dir = root_gs_model_dir / track_id
            gs_model_list = torch.load(gs_model_dir / "gt_gs_model_list.pt", map_location=self.tuner_device)
            self.gt_gs_model_list.extend(gs_model_list)
        
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
        self.all_model_list = []
        for track_id in track_ids:
            gs_model_dir = root_gs_model_dir / track_id
            gs_model_list = torch.load(gs_model_dir / "gs_model_list.pt", map_location=self.tuner_device)
            query_points = torch.load(gs_model_dir / "query_points.pt", map_location=self.tuner_device)
            transform_mat_neutral_pose = torch.load(
                gs_model_dir / "transform_mat_neutral_pose.pt", map_location=self.tuner_device
            )
            motion_seq = torch.load(gs_model_dir / "motion_seq.pt", map_location=self.tuner_device)
            shape_params = torch.from_numpy(np.load(gs_model_dir / "shape_params.npy")).unsqueeze(0).to(self.tuner_device)
            model = (gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_params)
            self.all_model_list.append(model)


    def _prepare_joined_inputs(self):
        train_fields = tuple(self.train_params)
        self.gs_model_list: List[GaussianAppOutput] = []
        self.query_points = None
        self.transform_mat_neutral_pose = None
        self.motion_seq = None
        self.shape_param = None
        self.body_pose_param: Optional[torch.nn.Parameter] = None

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

        # self._init_body_pose_parameter()

    def _init_body_pose_parameter(self):
        """
        Promote SMPL-X body pose tensor to a trainable parameter so it can
        receive gradients alongside the Gaussian attributes.
        """
        if self.motion_seq is None:
            return
        smplx_params = self.motion_seq["smplx_params"]
        body_pose = smplx_params.get("body_pose")
        if body_pose is None:
            return

        body_pose = body_pose.to(self.tuner_device)
        self.body_pose_param = torch.nn.Parameter(body_pose.detach().clone(), requires_grad=True)
        smplx_params["body_pose"] = self.body_pose_param

    # ---------------- Evaluation utilities ----------------
    def _load_gt_smplx_params(self, frame_paths: List[str], smplx_dir: Path):
        """Load per-frame SMPL-X params in world coordinates (no camera transform)."""

        npzs = [np.load(smplx_dir / f"{Path(fp).stem}.npz") for fp in frame_paths]

        def stack_key(key):
            arrs = [torch.from_numpy(n[key]).float() for n in npzs]
            return torch.stack(arrs, dim=1).to(self.tuner_device)  # [P, F, ...]

        betas = stack_key("betas")  # [P,F,10] (constant across frames)

        smplx = {
            "betas": betas[:, 0, :10],  # [P,10] keep first 10, assume constant over frames
            "root_pose": stack_key("root_pose"),   # [P,F,3] world axis-angle
            "body_pose": stack_key("body_pose"),
            "jaw_pose": stack_key("jaw_pose"),
            "leye_pose": stack_key("leye_pose"),
            "reye_pose": stack_key("reye_pose"),
            "lhand_pose": stack_key("lhand_pose"),
            "rhand_pose": stack_key("rhand_pose"),
            "trans": stack_key("trans"),           # [P,F,3] world translation
            "expr": stack_key("expression"),
            "transform_mat_neutral_pose": self.transform_mat_neutral_pose.to(self.tuner_device),
        }

        return smplx

    def _mask_centroids(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cx, cy, valid) for a batch of masks shaped [B, H, W]."""
        binary = (masks > 0.5).float()
        area = binary.flatten(1).sum(1)
        valid = area > 10.0  # ignore tiny masks
        if not valid.any():
            zeros = torch.zeros_like(area)
            return zeros, zeros, valid

        _, h, w = binary.shape
        xs = torch.arange(w, device=masks.device).view(1, 1, w)
        ys = torch.arange(h, device=masks.device).view(1, h, 1)
        cx = (binary * xs).sum((1, 2)) / (area + 1e-6)
        cy = (binary * ys).sum((1, 2)) / (area + 1e-6)
        return cx, cy, valid

    def _mask_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """Compute IoU between predicted and GT masks; expects shape [B, H, W, 1]."""
        p = (pred_mask[..., 0] > 0.5).float()
        g = (gt_mask[..., 0] > 0.5).float()
        inter = (p * g).sum((1, 2))
        union = (p + g - p * g).sum((1, 2)).clamp_min(1e-6)
        return inter / union

    def _estimate_world_translation_offset(
        self,
        render_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        intr: torch.Tensor,
        c2w: torch.Tensor,
        w2c: torch.Tensor,
        smplx_trans: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Estimate per-frame world-space translation that aligns rendered and GT masks.
        Returns tensor shaped [B, 3] or None if no valid masks.
        """
        # strip the channel dim -> [B, H, W]
        render_mask_b = render_mask[..., 0]
        gt_mask_b = gt_mask[..., 0]

        render_cx, render_cy, render_valid = self._mask_centroids(render_mask_b)
        gt_cx, gt_cy, gt_valid = self._mask_centroids(gt_mask_b)
        valid = render_valid & gt_valid
        if not valid.any():
            return None

        dx = gt_cx - render_cx
        dy = gt_cy - render_cy

        # Depth from SMPL-X translations (projected to camera space)
        cam_coords = torch.einsum("ij,pbj->pbi", w2c[:3, :3], smplx_trans) + w2c[:3, 3]
        z = cam_coords[..., 2].mean(dim=0)

        z = z.clamp_min(1e-3)
        fx, fy = intr[0, 0], intr[1, 1]
        delta_cam = torch.stack(
            [dx * z / fx, dy * z / fy, torch.zeros_like(z)],
            dim=-1,
        )
        delta_cam = torch.where(valid[:, None], delta_cam, torch.zeros_like(delta_cam))
        delta_world = torch.einsum("ij,bj->bi", c2w[:3, :3], delta_cam)
        return delta_world

    # ---------------- Training utilities ----------------
    def _trainable_tensors(self) -> List[torch.Tensor]:
        params = []
        for pidx, gauss in enumerate(self.gt_gs_model_list):
            print(f"For the {pidx}-th gauss model, adding trainable tensors:")
            for name in self.train_params:
                t = getattr(gauss, name, None)
                if torch.is_tensor(t):
                    t.requires_grad_(True)
                    print(f"    - {name}")
                    params.append(t)

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
        return smplx, render_c2ws, render_intrs

    def _slice_motion_using_gt(self, frame_indices: torch.Tensor):

        # Batched smplx params
        keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
            "expr",
        ]
        batch_smplx = dict()
        batch_smplx["transform_mat_neutral_pose"] = self.gt_smplx["transform_mat_neutral_pose"]
        batch_smplx["betas"] = self.gt_smplx["betas"]
        for key in keys:
            batch_smplx[key] = torch.index_select(
                self.gt_smplx[key], 1, frame_indices.to(self.gt_smplx[key].device)
            ).to(self.tuner_device)

        # Camera parameters
        bsize = frame_indices.shape[0]

        batch_c2w = self.c2w.to(self.tuner_device)          # [4, 4]
        batch_c2w = batch_c2w.unsqueeze(0).unsqueeze(1)     # [1, 1, 4, 4]
        batch_c2w = batch_c2w.expand(1, bsize, 4, 4)        # [1, bsize, 4, 4]

        batch_intr = self.K.to(self.tuner_device) # [4,4]
        batch_intr = batch_intr.unsqueeze(0).unsqueeze(1) # [1, 1, 4, 4]
        batch_intr = batch_intr.expand(1, bsize, 4, 4) # [1, bsize, 4, 4]

        return batch_smplx, batch_c2w, batch_intr

    def animation_infer_custom(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_hw):
        '''Inference code avoid repeat forward.
        '''

        # render target views
        render_res_list = []
        num_views = render_c2ws.shape[1]
        render_h, render_w = render_hw

        for view_idx in range(num_views):
            smplx_single_view = self.renderer.get_single_view_smpl_data(smplx_params, view_idx)
            render_res = self.renderer.forward_animate_gs_custom(
                gs_model_list,
                query_points,
                smplx_single_view,
                render_c2ws[:, view_idx : view_idx + 1],
                render_intrs[:, view_idx : view_idx + 1],
                render_h,
                render_w,
                render_bg_colors[:, view_idx : view_idx + 1],
            )
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                if isinstance(v[0], torch.Tensor):
                    out[k].append(v)
                else:
                    out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(
                        0, 2, 3, 1
                    )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
            else:
                out[k] = v
        return out

    def _render_batch(self, frame_indices: torch.Tensor):

        # Prepare rendering inputs
        smplx_params, render_c2ws, render_intrs = self._slice_motion_using_gt(frame_indices)

        # Override background to black
        render_bg_colors = torch.zeros((1, self.cfg.batch_size, 3), device=self.tuner_device)

        # Render
        return self.animation_infer_custom(
            self.gt_gs_model_list,
            self.gt_query_points,
            smplx_params,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
            render_hw=self.trn_render_hw,
        ), smplx_params, render_c2ws, render_intrs

    def _canonical_regularization(self):
        """Return combined canonical regularization and its components."""
        asap_terms = []
        acap_terms = []
        for gauss in self.gt_gs_model_list:
            # Gaussian Shape Regularization (ASAP): encourage isotropic scales.
            # If scaling is stored in log-space, exp() keeps positivity; otherwise it’s a smooth surrogate.
            scales = gauss.scaling
            scales_pos = torch.exp(scales)
            asap = ((scales_pos - scales_pos.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean()
            asap_terms.append(asap)

            # Positional anchoring (ACAP): hinge on offset magnitude beyond margin.
            offsets = gauss.offset_xyz
            acap = torch.clamp(offsets.norm(dim=-1) - self.cfg.regularization['acap_margin'], min=0.0).mean()
            acap_terms.append(acap)

            # Compute the percentage of Gaussians that are outside the margin
            with torch.no_grad():
                num_outside = (offsets.norm(dim=-1) > self.cfg.regularization['acap_margin']).sum().item()
                total = offsets.shape[0]
                percent_outside = (num_outside / total) * 100.0
                to_log = {"debug/acap_percent_outside": percent_outside}
                if self.wandb_run.enable:
                    wandb.log(to_log)

        asap_loss = torch.stack(asap_terms).mean() * self.cfg.regularization["asap_w"]
        acap_loss = torch.stack(acap_terms).mean() * self.cfg.regularization["acap_w"]

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

        # Initialize wandb if enabled and not already initialized
        if self.wandb_run is None:
            self._init_wandb()

        # Prepare dataset and dataloader
        dataset = FrameMaskDataset(
            self.frames_dir, self.masks_dir, self.tuner_device, sample_every=self.cfg.sample_every
        )
        first_frame = dataset[0][1]
        self.trn_render_hw = first_frame.shape[:2]
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Initialize optimizer
        params = self._trainable_tensors()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Pre-optimization visualization (epoch 0).
        if self.cfg.eval_pretrain:
            self.eval_loop(0)

        # Training loop
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            batch = 0
            for frame_indices, frames, masks, frame_paths in pbar:

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                frame_indices = frame_indices.to(self.tuner_device)
                res, smplx_params, render_c2ws, render_intrs = self._render_batch(frame_indices)
                pred_rgb = res["comp_rgb"]  # [B, H, W, 3], 0-1
                pred_mask = res["comp_mask"][..., :1]  # [B, H, W, 1]

                # Compute masked ground truth
                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3

                # BBOX crop around the union person mask
                if self.cfg.use_bbox_crop:
                    bboxes = get_masks_based_bbox(masks, pad=self.cfg.bbox_pad)
                    loss_pred_rgb = bbox_crop(bboxes, pred_rgb)
                    loss_pred_mask = bbox_crop(bboxes, pred_mask)
                    loss_gt_masked = bbox_crop(bboxes, gt_masked)
                    loss_masks = bbox_crop(bboxes, masks)
                else:
                    loss_pred_rgb = pred_rgb
                    loss_pred_mask = pred_mask
                    loss_gt_masked = gt_masked
                    loss_masks = masks

                # Compute loss
                rgb_loss = self.cfg.loss_weights["rgb"] * F.mse_loss(loss_pred_rgb, loss_gt_masked)
                sil_loss = self.cfg.loss_weights["sil"] * F.mse_loss(loss_pred_mask, loss_masks)
                ssim_val = fused_ssim(_ensure_nchw(loss_pred_rgb), _ensure_nchw(loss_gt_masked), padding="valid")
                ssim_loss = self.cfg.loss_weights["ssim"] * (1.0 - ssim_val)
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = rgb_loss + sil_loss + ssim_loss + reg_loss
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                optimizer.step()

                # Log the results
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
                            "loss/ssim": ssim_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                        }
                    )


                batch += 1

            # End of epoch
            # - Report average loss
            avg_loss = running_loss / max(1, len(loader))
            if self.wandb_run is not None:
                wandb.log({"loss/combined_epoch": avg_loss, "epoch": epoch + 1})

            # - Run eval loop if neccesary
            if getattr(self.cfg, "eval_every_epoch", 0) > 0 and (epoch + 1) % self.cfg.eval_every_epoch == 0:
                self.eval_loop(epoch + 1)
            
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ---------------- Evaluation -------------------
    @torch.no_grad()
    def eval_loop(self, epoch):

        # Parse the evaluation setup
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)
        camera_params_path: Path = root_gt_dir_path / "cameras" / "rgb_cameras.npz"
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)
        num_tracks, num_frames = self.cfg.num_persons, self.cfg.batch_size
        render_bg_colors_template = torch.zeros(
            (num_tracks, num_frames, 3), device=self.tuner_device, dtype=torch.float32
        )  # black
        alpha_candidates = torch.arange(-0.9, 1.1, 0.1, device=self.tuner_device)
        max_shift_m = 0.2

        # Collector for metrics
        metrics_all_cams_per_frame = list()

        for tgt_cam_id in target_camera_ids:

            # Prepare paths
            tgt_gt_frames_dir_path = root_gt_dir_path / "images" / f"{tgt_cam_id}"
            tgt_gt_masks_dir_path = root_gt_dir_path / "seg" / "img_seg_mask" / f"{tgt_cam_id}" / "all"

            # Load and prepare the gt camera parameters
            # - c2w
            tgt_intr, tgt_extr = load_camera_from_npz(camera_params_path, tgt_cam_id, device=self.tuner_device)
            tgt_w2c = extr_to_w2c_4x4(tgt_extr, self.tuner_device)
            tgt_c2w = torch.inverse(tgt_w2c)
            tgt_intr4 = intr_to_4x4(tgt_intr, self.tuner_device)

            # Prepare dataset and dataloader for loading frames and masks
            dataset = FrameMaskDataset(tgt_gt_frames_dir_path, tgt_gt_masks_dir_path, self.tuner_device, sample_every=1)
            loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

            # Prepare paths where to save results
            save_dir = root_save_dir / f"{tgt_cam_id}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Render in batches novel views from target camera
            with torch.no_grad():
                for frame_indices, frames, masks, frame_paths in tqdm(loader, desc=f"NVS cam {tgt_cam_id}", leave=False):

                    # Get gt image size
                    gt_h, gt_w = frames.shape[1], frames.shape[2]
                    batch_b = frames.shape[0]

                    # Prepare per-batch camera/intr/background tensors (num_frames can differ in last batch)
                    render_c2ws = (
                        tgt_c2w.unsqueeze(0).unsqueeze(0).expand(num_tracks, batch_b, 4, 4).clone()
                    )
                    render_intrs = tgt_intr4.unsqueeze(0).unsqueeze(0).expand(num_tracks, batch_b, 4, 4)
                    render_bg_colors = render_bg_colors_template[:, :batch_b]
                    
                    # Load the gt SMPL-X parameters
                    frame_indices = frame_indices.to(self.tuner_device)
                    smplx_params, _, _ = self._slice_motion_using_gt(frame_indices)

                    # Render with the model
                    res = self.animation_infer_custom(
                        self.gt_gs_model_list,
                        self.gt_query_points,
                        smplx_params,
                        render_c2ws=render_c2ws,
                        render_intrs=render_intrs,
                        render_bg_colors=render_bg_colors,
                        render_hw=(gt_h, gt_w),
                    )

#                    # Estimate translation offset between rendered and GT masks
                    #delta_world = self._estimate_world_translation_offset(
                        #render_mask=res["comp_rgb"] > 0.05,
                        #gt_mask=masks,
                        #intr=tgt_intr,
                        #c2w=tgt_c2w,
                        #w2c=tgt_w2c,
                        #smplx_trans=smplx_params["trans"],
                    #)
                    #delta_world = delta_world.clamp(-max_shift_m, max_shift_m)

                    ## Find the best alpha for the translation offset
                    #best_res = res
                    #best_iou = -1.0
                    #for idx, alpha in enumerate(alpha_candidates):
                        #cand_c2ws = render_c2ws.clone()
                        #cand_c2ws[:, :, :3, 3] -= alpha * delta_world.unsqueeze(0)
                        #cand_res = self.animation_infer_custom(
                            #self.gs_model_list,
                            #self.query_points,
                            #smplx_params,
                            #render_c2ws=cand_c2ws,
                            #render_intrs=render_intrs,
                            #render_bg_colors=render_bg_colors,
                            #render_hw=(gt_h, gt_w),
                        #)
                        #iou = self._mask_iou(cand_res["comp_rgb"] > 0.05, masks).mean()
                        #if iou > best_iou:
                            #best_iou = iou
                            #best_res = cand_res
                    #res = best_res

                    # Save rendered images
                    renders = res["comp_rgb"]  # [B, H, W, 3]

                    # Apply masks to the gt 
                    masks3 = masks.repeat(1, 1, 1, 3)
                    masked_gt = frames * masks3

                    # Combine masked render and masked GT with overlay
                    render_vs_gt_dir = save_dir / "render_vs_gt"
                    render_vs_gt_dir.mkdir(parents=True, exist_ok=True)
                    columns = [renders, masked_gt]
                    joined = torch.cat(columns, dim=2)  # side-by-side along width
                    for i in range(joined.shape[0]):
                        save_path = render_vs_gt_dir / Path(frame_paths[i]).name
                        save_image(joined[i].permute(2, 0, 1), str(save_path))

                    # Before evaluation, apply downsampling if needed
                    if self.cfg.nvs_eval.downscale_factor > 1:
                        ds_factor = self.cfg.nvs_eval.downscale_factor
                        renders = F.interpolate(
                            renders.permute(0, 3, 1, 2),
                            size=(gt_h // ds_factor, gt_w // ds_factor),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                        frames = F.interpolate(
                            frames.permute(0, 3, 1, 2),
                            size=(gt_h // ds_factor, gt_w // ds_factor),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                        masks = F.interpolate(
                            masks.permute(0, 3, 1, 2),
                            size=(gt_h // ds_factor, gt_w // ds_factor),
                            mode="nearest",
                        ).permute(0, 2, 3, 1)
                        masks3 = masks.repeat(1, 1, 1, 3)
                        frames = frames * masks3
                        
                        masks = torch.ones_like(masks)
                        masks3 = masks.repeat(1, 1, 1, 3)
                    
                    # Save what goes into metrics computation for debugging
                    metrics_debug_dir = save_dir / "metrics_debug_inputs"
                    metrics_debug_dir.mkdir(parents=True, exist_ok=True)
                    masked_renders = renders * masks3
                    masked_gt = frames * masks3
                    overlayed = 0.5 * masked_renders + 0.5 * masked_gt
                    columns = [masked_renders, masked_gt, overlayed]
                    joined = torch.cat(columns, dim=2)  # side-by-side along width
                    for i in range(joined.shape[0]):
                        save_path = metrics_debug_dir / Path(frame_paths[i]).name
                        save_image(joined[i].permute(2, 0, 1), str(save_path))
                    
                    # Compute metrics
                    psnr_vals = psnr(
                        images=frames,
                        masks=masks.squeeze(-1),
                        renders=renders,
                    )
                    ssim_vals = ssim(
                        images=frames,
                        masks=masks.squeeze(-1),
                        renders=renders,
                    )
                    lpips_vals = lpips(
                        images=frames,
                        masks=masks.squeeze(-1),
                        renders=renders,
                    )
                    for fid, psnr_val, ssim_val, lpips_val in zip(frame_indices, psnr_vals, ssim_vals, lpips_vals):
                        metrics_all_cams_per_frame.append(
                            (tgt_cam_id, fid.item(), psnr_val.item(), ssim_val.item(), lpips_val.item())
                        )
            
            # Create a video from render vs gt images using call to ffmpeg
            render_vs_gt_dir = root_save_dir / f"{tgt_cam_id}" / "render_vs_gt"
            video_path = root_save_dir / f"{tgt_cam_id}" / f"cam_{tgt_cam_id}_nvs_epoch_{epoch:04d}.mp4"
            jpg_frames = sorted(p for p in render_vs_gt_dir.glob("*.jpg") if p.stem.isdigit())
            if not jpg_frames:
                raise FileNotFoundError(f"No render_vs_gt frames found in {render_vs_gt_dir}")
            start_number = int(jpg_frames[0].stem)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-start_number", str(start_number),
                "-framerate", "20",
                "-i", str(render_vs_gt_dir / "%06d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
 
        # Save metrics to CSV
        # - save per-frame metrics
        df = pd.DataFrame(metrics_all_cams_per_frame, columns=["camera_id", "frame_id", "psnr", "ssim", "lpips"])
        csv_path = root_save_dir / "metrics_all_cams_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics per camera
        df_avg_cam = df.groupby("camera_id").agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"}).reset_index()
        csv_path_avg_cam = root_save_dir / "metrics_avg_per_camera.csv"
        df_avg_cam.to_csv(csv_path_avg_cam, index=False)

        # - log to wandb the per cam metrics
        if self.cfg.wandb.enable:
            for _, row in df_avg_cam.iterrows():
                to_log = {f"eval_nv/cam_{int(row['camera_id'])}/{metric_name}": row[metric_name] for metric_name in ["psnr", "ssim", "lpips"]}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # - save overall average metrics excluding self.cfg.nvs_eval.source_camera_id
        df_excluding_source = df[df["camera_id"] != self.cfg.nvs_eval.source_camera_id]
        overall_avg = {
            "psnr": df_excluding_source["psnr"].mean(),
            "ssim": df_excluding_source["ssim"].mean(),
            "lpips": df_excluding_source["lpips"].mean(),
        }
        overall_avg_path = root_save_dir / "novel_view_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")

        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_nv/all_cam/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    tuner = MultiHumanTrainer(cfg)
    tuner.train_loop()


if __name__ == "__main__":
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
