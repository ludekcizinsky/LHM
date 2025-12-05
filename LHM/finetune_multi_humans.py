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
# Utility functions
# ---------------------------------------------------------------------------
import math

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


def smplx_base_vertices_in_camera(
    smplx_model,
    smplx_params: dict,
    pid: int,
    frame_idx: int,
    c2w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute base (non-upsampled) SMPL-X vertices in camera coordinates for a given person and frame.
    Returns: [V_base, 3] or None on failure.
    """
    try:
        def _pad_or_truncate(t: torch.Tensor, target_dim: int | None, label: str) -> torch.Tensor:
            if target_dim is None:
                return t
            cur = t.shape[-1]
            if cur == target_dim:
                return t
            if cur > target_dim:
                print(f"[DEBUG] Truncating {label} from {cur} to {target_dim}")
                return t[..., :target_dim]
            pad = torch.zeros(*t.shape[:-1], target_dim - cur, device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=-1)

        layer = getattr(smplx_model, "smplx_layer", None)
        if layer is None and hasattr(smplx_model, "layer"):
            layer = smplx_model.layer.get("neutral", None)
        if layer is None:
            raise AttributeError("SMPLX model has no base smplx_layer")
        layer = layer.to(device)

        expected_beta_dim = getattr(layer, "num_betas", None)
        if expected_beta_dim is None and hasattr(layer, "shapedirs"):
            try:
                expected_beta_dim = int(layer.shapedirs.shape[-1])
            except Exception:
                expected_beta_dim = None
        expected_expr_dim = getattr(layer, "num_expression_coeffs", None)
        if expected_expr_dim is None and hasattr(layer, "expr_dirs"):
            try:
                expected_expr_dim = int(layer.expr_dirs.shape[-1])
            except Exception:
                expected_expr_dim = None

        params = {
            "global_orient": smplx_params["root_pose"][pid : pid + 1, frame_idx],
            "body_pose": smplx_params["body_pose"][pid : pid + 1, frame_idx],
            "jaw_pose": smplx_params["jaw_pose"][pid : pid + 1, frame_idx],
            "leye_pose": smplx_params["leye_pose"][pid : pid + 1, frame_idx],
            "reye_pose": smplx_params["reye_pose"][pid : pid + 1, frame_idx],
            "left_hand_pose": smplx_params["lhand_pose"][pid : pid + 1, frame_idx],
            "right_hand_pose": smplx_params["rhand_pose"][pid : pid + 1, frame_idx],
            "betas": _pad_or_truncate(smplx_params["betas"][pid : pid + 1], expected_beta_dim, "betas"),
            "transl": smplx_params["trans"][pid : pid + 1, frame_idx],
        }
        if "expr" in smplx_params:
            expr = smplx_params["expr"][pid : pid + 1, frame_idx]
            params["expression"] = _pad_or_truncate(expr, expected_expr_dim, "expr")
        output = layer(**{k: v.to(device) for k, v in params.items()})
        verts_world = output.vertices  # [1, V, 3]
        w2c = torch.inverse(c2w).to(device)
        homo = torch.cat([verts_world, torch.ones_like(verts_world[..., :1])], dim=-1)  # [1,V,4]
        cam = (w2c @ homo.transpose(1, 2)).transpose(1, 2)[0, :, :3]  # [V,3]
        return cam
    except Exception as e:
        print(f"[DEBUG] Could not compute base SMPL-X verts in camera: {e}")
        return None


SMPLX_BODY_TO_SMPL = list(range(22))
LEFT_PALM_IDS = [25, 28, 34, 31, 37]
RIGHT_PALM_IDS = [40, 43, 49, 46, 52]

# Optional: paths to SMPLX→SMPL transfer and SMPL joint regressor
SMPLX2SMPL_TRANSFER_PATH = Path("/scratch/izar/cizinsky/pretrained/smplx2smpl.pkl")
SMPL_JOINT_REGRESSOR_PATH = Path("/home/cizinsky/body_models/smpl/SMPL_NEUTRAL.pkl")

_SMPL_REGRESSOR_CACHE: torch.Tensor | None = None
_SMPLX2SMPL_TRANSFER_CACHE: torch.Tensor | None = None
_WARNED_TRANSFER_SHAPE: bool = False
_SMPL_FACE_CACHE: np.ndarray | None = None


def _load_smpl_regressor() -> torch.Tensor | None:
    global _SMPL_REGRESSOR_CACHE
    if _SMPL_REGRESSOR_CACHE is not None:
        return _SMPL_REGRESSOR_CACHE
    try:
        import pickle
        with open(SMPL_JOINT_REGRESSOR_PATH, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        J = data.get("J_regressor", None)
        if J is None:
            return None
        if hasattr(J, "toarray"):
            J = J.toarray()
        J_torch = torch.from_numpy(J).float()  # [24, 6890]
        _SMPL_REGRESSOR_CACHE = J_torch
        return J_torch
    except Exception as e:
        print(f"[WARN] Could not load SMPL joint regressor: {e}")
        return None


def _load_smplx2smpl_transfer() -> torch.Tensor | None:
    global _SMPLX2SMPL_TRANSFER_CACHE
    if _SMPLX2SMPL_TRANSFER_CACHE is not None:
        return _SMPLX2SMPL_TRANSFER_CACHE
    try:
        import pickle
        with open(SMPLX2SMPL_TRANSFER_PATH, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        mat = data.get("matrix", None)  # expected [6890, 10475]
        if mat is None:
            return None
        mat_torch = torch.from_numpy(mat).float()
        _SMPLX2SMPL_TRANSFER_CACHE = mat_torch
        return mat_torch
    except Exception as e:
        print(f"[WARN] Could not load SMPLX→SMPL transfer: {e}")
        return None


def _load_smpl_faces() -> np.ndarray | None:
    global _SMPL_FACE_CACHE
    if _SMPL_FACE_CACHE is not None:
        return _SMPL_FACE_CACHE
    try:
        import pickle
        with open(SMPL_JOINT_REGRESSOR_PATH, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        faces = data.get("f", None)
        if faces is None:
            return None
        faces = np.asarray(faces, dtype=np.int64)
        _SMPL_FACE_CACHE = faces
        return faces
    except Exception as e:
        print(f"[WARN] Could not load SMPL faces: {e}")
        return None


def smplx_to_smpl(joints55: torch.Tensor | None, smplx_vertices: torch.Tensor | None = None) -> torch.Tensor:
    """
    Convert SMPL-X joints [...,55,3] to SMPL 24-joint layout by copying body joints
    and averaging finger bases for the hand markers.
    If SMPL-X vertices and the SMPL joint regressor + transfer are available,
    regress SMPL joints from SMPL vertices for higher fidelity.
    """
    # Robust path: regress from vertices if possible
    if smplx_vertices is not None:
        reg = _load_smpl_regressor()
        transfer = _load_smplx2smpl_transfer()
        if reg is not None and transfer is not None:
            try:
                global _WARNED_TRANSFER_SHAPE
                if smplx_vertices.shape[-2] != transfer.shape[1]:
                    if not _WARNED_TRANSFER_SHAPE:
                        print(
                            f"[WARN] SMPLX verts ({smplx_vertices.shape[-2]}) != transfer cols ({transfer.shape[1]}), "
                            "skipping regressed SMPL joints."
                        )
                        _WARNED_TRANSFER_SHAPE = True
                    raise ValueError("transfer/verts shape mismatch")
                # transfer: [6890, 10475], smplx_vertices: [..., 10475, 3]

                smpl_vertices = torch.einsum(
                    "nv,...vc->...nc",
                    transfer.to(smplx_vertices.device, smplx_vertices.dtype),
                    smplx_vertices,
                )  # [..., 6890, 3]
                smpl_joints = torch.einsum(
                    "jn,...nc->...jc",
                    reg.to(smpl_vertices.device, smpl_vertices.dtype),
                    smpl_vertices,
                )  # [..., 24, 3]
                return smpl_joints
            except Exception as e:
                print(f"[WARN] Falling back to heuristic SMPLX→SMPL joints: {e}")

    # Heuristic fallback (fast, no verts)
    if joints55 is None:
        raise ValueError("joints55 is required when SMPL-X vertices are not provided.")
    body = joints55[..., SMPLX_BODY_TO_SMPL, :]  # [...,22,3]
    l_hand = joints55[..., LEFT_PALM_IDS, :].mean(dim=-2, keepdim=True)
    r_hand = joints55[..., RIGHT_PALM_IDS, :].mean(dim=-2, keepdim=True)
    return torch.cat([body, l_hand, r_hand], dim=-2)  # [...,24,3]

def similarity_transform_3D(src, dst):
    """
    Compute the rotation R and translation t that aligns src to dst
    such that dst ≈ src @ R.T + t

    src, dst: (N, 3) arrays of corresponding 3D points
    returns: s, R (3x3), t (3,), avg_reproj_error
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    # 1. Compute centroids
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    # 2. Center the points
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    # 3. Compute covariance matrix
    H = src_centered.T @ dst_centered  # (3x3)

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute rotation
    R = Vt.T @ U.T

    # Reflection correction (ensure det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute scale
    var_src = np.sum(np.sum(src_centered**2, axis=1))
    s = np.sum(S) / var_src

    # 7. Compute translation
    t = centroid_dst - s*(centroid_src @ R.T)  

    # 8. compute reprojection error per point
    avg_reproj_error = 0.0
    for i in range(src.shape[0]):
        src_pt = src[i]
        dst_pt = dst[i]
        src_transformed = s * (R @ src_pt) + t
        error = np.linalg.norm(dst_pt - src_transformed)
        avg_reproj_error += error
    
    avg_reproj_error *= 1.0 / src.shape[0] 
    return s, R, t.squeeze(), avg_reproj_error

def compute_S_M2W_from_joints(pred_smpl_joints, gt_smpl_joints, joint_indices=None):
    """
    pred_smpl_joints: (F, P, J, 3) - Multi-HMR joints in M-frame
    gt_smpl_joints:   (F, P, J, 3) - GT joints in W-frame
    joint_indices: list/array of joint indices to use (in SMPL joint space).
                   If None, use a default core-body set.
    
    Returns:
        S_M2W: (4, 4) similarity transform matrix mapping M-frame -> W-frame
        s:     scalar scale
        R:     (3,3) rotation
        t:     (3,) translation
        avg_reproj_error: float
    """
    pred_smpl_joints = np.asarray(pred_smpl_joints.cpu())
    gt_smpl_joints   = np.asarray(gt_smpl_joints.cpu())
    assert pred_smpl_joints.shape == gt_smpl_joints.shape
    assert pred_smpl_joints.shape[-1] == 3

    F, P, J, _ = pred_smpl_joints.shape

    # Default: core body joints (example indices; adjust to your SMPL convention)
    # Here: 0 pelvis, 1 l_hip, 2 r_hip, 3 spine, 4 l_knee, 5 r_knee,
    #       6 spine1, 9 l_shoulder, 12 r_shoulder, 16 l_elbow, 17 r_elbow
    if joint_indices is None:
        joint_indices = [0, 1, 2, 3, 4, 5, 6, 9, 12, 16, 17]

    joint_indices = np.asarray(joint_indices, dtype=int)

    # Select joints
    pred_sel = pred_smpl_joints[:, :, :, :]  # (F, P, K, 3)
    gt_sel   = gt_smpl_joints[:, :, :, :]    # (F, P, K, 3)

    # Flatten over frames and people and joints -> (N, 3)
    pred_flat = pred_sel.reshape(-1, 3)
    gt_flat   = gt_sel.reshape(-1, 3)

    # Optionally: remove NaNs / infs
    valid_mask = np.isfinite(pred_flat).all(axis=1) & np.isfinite(gt_flat).all(axis=1)
    pred_flat = pred_flat[valid_mask]
    gt_flat   = gt_flat[valid_mask]

    # Compute similarity transform: pred (src, M-frame) -> gt (dst, W-frame)
    s, R, t, avg_err = similarity_transform_3D(pred_flat, gt_flat)

    # Build 4x4 S_M2W
    S_M2W = np.eye(4, dtype=np.float64)
    S_M2W[:3, :3] = s * R
    S_M2W[:3, 3]  = t

    return S_M2W, s, R, t, avg_err


def smplx_joints_in_camera(
    smplx_model,
    smplx_params: dict,
    pid: int,
    frame_idx: int,
    c2w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute SMPL-X joints (55) in camera coordinates for a given person and frame.
    Returns: [N, 3] or None on failure.
    """

    smpl_slice = {
        "betas": smplx_params["betas"][pid : pid + 1],
        "root_pose": smplx_params["root_pose"][pid : pid + 1, frame_idx],
        "body_pose": smplx_params["body_pose"][pid : pid + 1, frame_idx],
        "jaw_pose": smplx_params["jaw_pose"][pid : pid + 1, frame_idx],
        "leye_pose": smplx_params["leye_pose"][pid : pid + 1, frame_idx],
        "reye_pose": smplx_params["reye_pose"][pid : pid + 1, frame_idx],
        "lhand_pose": smplx_params["lhand_pose"][pid : pid + 1, frame_idx],
        "rhand_pose": smplx_params["rhand_pose"][pid : pid + 1, frame_idx],
        "trans": smplx_params["trans"][pid : pid + 1, frame_idx],
        "transform_mat_neutral_pose": smplx_params["transform_mat_neutral_pose"][pid : pid + 1],
    }
    if "expr" in smplx_params:
        smpl_slice["expr"] = smplx_params["expr"][pid : pid + 1, frame_idx]
    for opt_key in ("face_offset", "joint_offset", "locator_offset"):
        if opt_key in smplx_params:
            smpl_slice[opt_key] = smplx_params[opt_key][pid : pid + 1]

    # Drop any time dimension from betas (should be [1, num_betas])
    if smpl_slice["betas"].dim() > 2:
        smpl_slice["betas"] = smpl_slice["betas"][:, 0]

    joint_zero = smplx_model.get_zero_pose_human(
        shape_param=smpl_slice["betas"],
        device=device,
        face_offset=smpl_slice.get("face_offset", None),
        joint_offset=smpl_slice.get("joint_offset", None),
        return_mesh=False,
    )
    _, posed_joints = smplx_model.get_transform_mat_joint(
        smpl_slice["transform_mat_neutral_pose"], joint_zero, smpl_slice
    )
    joints_world = posed_joints + smpl_slice["trans"]  # [1,55,3]
    w2c = torch.inverse(c2w).to(device)
    homo = torch.cat([joints_world, torch.ones_like(joints_world[..., :1])], dim=-1)  # [1,55,4]
    cam = (w2c @ homo.transpose(1, 2)).transpose(1, 2)[0, :, :3]  # [55,3]
    return cam


def load_gt_smpl_joints_cam(
    smpl_dir: Path,
    frame_path: str | Path,
    person_idx: int,
    c2w: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Load GT SMPL joints (24) for a given frame/person and return camera-space joints [24,3].
    c2w: [4,4] camera-to-world (will be inverted internally).
    """
    try:
        stem = Path(frame_path).stem
        npz_path = smpl_dir / f"{stem}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        joints_np = data["joints_3d"]  # [P,24,3]
        if person_idx >= joints_np.shape[0]:
            return None
        joints = torch.from_numpy(joints_np[person_idx]).to(device=device, dtype=torch.float32)  # [24,3]
        w2c = torch.inverse(c2w.to(device))
        homo = torch.cat([joints, torch.ones_like(joints[:, :1])], dim=-1)  # [24,4]
        cam = (w2c @ homo.transpose(0, 1)).transpose(0, 1)[..., :3]  # [24,3]
        return cam
    except Exception as e:
        print(f"[DEBUG] Could not load GT SMPL joints cam for {frame_path}: {e}")
        return None



def overlay_smpl_joints(
        images: torch.Tensor,
        joints: torch.Tensor,
        intrs: torch.Tensor,
        joint_radius: int = 3,
        color: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),
        device: torch.device = torch.device("cpu"),
):

    updated = []
    F, P = joints.shape[0], joints.shape[1]
    color = color.to(device)
    intr = intrs[:3, :3].to(device)

    for fi in range(F):
        gt_img = images[fi].clone()
        H, W = gt_img.shape[0], gt_img.shape[1]

        for person_idx in range(P):
            cam_joints = joints[fi, person_idx]

            cam_z = cam_joints[:, 2].clamp(min=1e-6)
            uvx = (intr[0, 0] * cam_joints[:, 0] + intr[0, 2] * cam_z) / cam_z
            uvy = (intr[1, 1] * cam_joints[:, 1] + intr[1, 2] * cam_z) / cam_z
            u = uvx.round().long()
            v = uvy.round().long()
            body_joint_ids = list(range(min(24, u.shape[0])))  # SMPL has 24 body joints
            for ui, vi in zip(u[body_joint_ids].tolist(), v[body_joint_ids].tolist()):
                if 0 <= ui < W and 0 <= vi < H:
                    v0 = max(vi - joint_radius, 0)
                    v1 = min(vi + joint_radius + 1, H)
                    u0 = max(ui - joint_radius, 0)
                    u1 = min(ui + joint_radius + 1, W)
                    gt_img[v0:v1, u0:u1, :] = color

        updated.append(gt_img)

    return torch.stack(updated, dim=0)


def overlay_smplx_mesh_pyrender(
    images: torch.Tensor,
    smplx_params: dict,
    smplx_model,
    intr: torch.Tensor,
    c2w: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    mesh_alpha: float = 0.7,
) -> torch.Tensor:
    """
    Render SMPL-X meshes with trimesh+pyrender and alpha-blend them over images.

    images: [F, H, W, 3] float in [0,1]
    smplx_params: dict with shapes [P, F, ...]
    intr: [3,3] or [4,4] intrinsics
    c2w: [4,4] camera-to-world
    mesh_color: RGB in [0,1]; mesh_alpha: opacity for the mesh layer
    """
    try:
        import pyrender
        import trimesh
    except Exception as e:
        print(f"[WARN] pyrender/trimesh unavailable for mesh overlay: {e}")
        return images

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    layer = getattr(smplx_model, "smplx_layer", None)
    if layer is None and hasattr(smplx_model, "layer"):
        layer = smplx_model.layer.get("neutral", None)
    faces = getattr(layer, "faces", None) if layer is not None else None
    if faces is None:
        print("[WARN] SMPL-X faces not found, skipping mesh overlay.")
        return images
    faces_np = np.asarray(faces, dtype=np.int64)

    intr_cpu = intr.detach().cpu()
    if intr_cpu.shape[-2:] == (4, 4):
        intr_cpu = intr_cpu[:3, :3]
    fx, fy, cx, cy = (
        float(intr_cpu[0, 0]),
        float(intr_cpu[1, 1]),
        float(intr_cpu[0, 2]),
        float(intr_cpu[1, 2]),
    )

    num_frames = images.shape[0]
    num_people = smplx_params["betas"].shape[0]
    H, W = images.shape[1], images.shape[2]

    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    except Exception as e:
        print(f"[WARN] Could not initialise pyrender renderer: {e}")
        return images

    out_frames: List[torch.Tensor] = []
    try:
        for fi in range(num_frames):
            base_img = (images[fi].detach().cpu().numpy() * 255).astype(np.uint8)
            depth_map = np.ones((H, W)) * np.inf
            overlay_img = base_img.astype(np.float32)

            for pid in range(num_people):
                cam_verts = smplx_base_vertices_in_camera(
                    smplx_model, smplx_params, pid, fi, c2w, device
                )
                if cam_verts is None:
                    continue
                verts_np = cam_verts.detach().cpu().numpy()

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode="BLEND",
                    baseColorFactor=[
                        float(mesh_color[0]),
                        float(mesh_color[1]),
                        float(mesh_color[2]),
                        float(mesh_alpha),
                    ],
                )
                mesh = trimesh.Trimesh(verts_np, faces_np, process=False)
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5)
                )
                scene.add(mesh, "mesh")
                camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
                scene.add(camera, pose=np.eye(4))
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                scene.add(light, pose=np.eye(4))

                color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                valid_mask = (rend_depth < depth_map) & (rend_depth > 0)
                depth_map[valid_mask] = rend_depth[valid_mask]
                valid_mask = valid_mask[..., None]
                overlay_img = valid_mask * color[..., :3] + (1.0 - valid_mask) * overlay_img

            overlay_tensor = (
                torch.from_numpy(overlay_img).to(device=images.device, dtype=images.dtype) / 255.0
            )
            out_frames.append(overlay_tensor)
    finally:
        renderer.delete()

    return torch.stack(out_frames, dim=0)


def overlay_smpl_mesh_pyrender(
    images: torch.Tensor,
    smpl_verts_world: torch.Tensor,
    intr: torch.Tensor,
    c2w: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.0, 0.5, 1.0),
    mesh_alpha: float = 0.7,
) -> torch.Tensor:
    """
    Render SMPL meshes (world-space verts) with trimesh+pyrender and alpha-blend them over images.

    images: [F, H, W, 3] float in [0,1]
    smpl_verts_world: [F, P, V, 3] world-space vertices
    intr: [3,3] or [4,4] intrinsics
    c2w: [4,4] camera-to-world
    """
    faces = _load_smpl_faces()
    if faces is None:
        print("[WARN] SMPL faces not available, skipping SMPL overlay.")
        return images

    try:
        import pyrender
        import trimesh
    except Exception as e:
        print(f"[WARN] pyrender/trimesh unavailable for SMPL overlay: {e}")
        return images

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    intr_cpu = intr.detach().cpu()
    if intr_cpu.shape[-2:] == (4, 4):
        intr_cpu = intr_cpu[:3, :3]
    fx, fy, cx, cy = (
        float(intr_cpu[0, 0]),
        float(intr_cpu[1, 1]),
        float(intr_cpu[0, 2]),
        float(intr_cpu[1, 2]),
    )

    num_frames = images.shape[0]
    num_people = smpl_verts_world.shape[1]
    H, W = images.shape[1], images.shape[2]
    w2c = torch.inverse(c2w.to(device))  # [4,4]

    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    except Exception as e:
        print(f"[WARN] Could not initialise pyrender renderer for SMPL overlay: {e}")
        return images

    out_frames: List[torch.Tensor] = []
    try:
        for fi in range(num_frames):
            base_img = (images[fi].detach().cpu().numpy() * 255).astype(np.uint8)
            depth_map = np.ones((H, W)) * np.inf
            overlay_img = base_img.astype(np.float32)

            for pid in range(num_people):
                verts_w = smpl_verts_world[fi, pid].to(device=device)
                if verts_w.numel() == 0:
                    continue
                homo = torch.cat([verts_w, torch.ones_like(verts_w[:, :1])], dim=-1)  # [V,4]
                cam = (w2c @ homo.t()).t()[:, :3]
                verts_np = cam.detach().cpu().numpy()

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode="BLEND",
                    baseColorFactor=[
                        float(mesh_color[0]),
                        float(mesh_color[1]),
                        float(mesh_color[2]),
                        float(mesh_alpha),
                    ],
                )
                mesh = trimesh.Trimesh(verts_np, faces, process=False)
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5)
                )
                scene.add(mesh, "mesh")
                camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
                scene.add(camera, pose=np.eye(4))
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                scene.add(light, pose=np.eye(4))

                color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                valid_mask = (rend_depth < depth_map) & (rend_depth > 0)
                depth_map[valid_mask] = rend_depth[valid_mask]
                valid_mask = valid_mask[..., None]
                overlay_img = valid_mask * color[..., :3] + (1.0 - valid_mask) * overlay_img

            overlay_tensor = (
                torch.from_numpy(overlay_img).to(device=images.device, dtype=images.dtype) / 255.0
            )
            out_frames.append(overlay_tensor)
    finally:
        renderer.delete()

    return torch.stack(out_frames, dim=0)


def preload_gt_smpl_joints_camera_space(
    frame_paths: List[str | Path],
    smpl_dir: Path,
    c2w: torch.Tensor,
    num_persons: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load GT SMPL joints from npz and transform to camera space for the first `preload_frames` frames.
    Returns a tensor [F, P, 24, 3] 
    """
    smpl_joints = []
    for fp in frame_paths:
        per_person = []
        for pid in range(num_persons):
            cam_j = load_gt_smpl_joints_cam(smpl_dir, fp, pid, c2w, device)
            per_person.append(cam_j)
        smpl_joints.append(torch.stack(per_person, dim=0))

    return torch.stack(smpl_joints, dim=0)


def load_gt_smpl_joints_world(
    smpl_dir: Path,
    frame_path: str | Path,
    person_idx: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Load GT SMPL joints (24) for a given frame/person in world space.
    """
    try:
        stem = Path(frame_path).stem
        npz_path = smpl_dir / f"{stem}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        joints_np = data["joints_3d"]  # [P,24,3]
        if person_idx >= joints_np.shape[0]:
            return None
        joints = torch.from_numpy(joints_np[person_idx]).to(device=device, dtype=torch.float32)  # [24,3]
        return joints
    except Exception as e:
        print(f"[DEBUG] Could not load GT SMPL joints world for {frame_path}: {e}")
        return None


def load_gt_smpl_verts_world_batch(
    frame_paths: List[str | Path],
    smpl_dir: Path,
    num_persons: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Load GT SMPL verts (world space) for each frame. Returns [F, P, V, 3] or None if unavailable.
    """
    verts_list = []
    for fp in frame_paths:
        try:
            stem = Path(fp).stem
            npz_path = smpl_dir / f"{stem}.npz"
            if not npz_path.exists():
                return None
            data = np.load(npz_path)
            verts_np = data.get("verts", None)
            if verts_np is None:
                return None
            verts_t = torch.from_numpy(verts_np).float().to(device)  # [P,6890,3]
            if verts_t.shape[0] < num_persons:
                pad = torch.zeros(
                    (num_persons - verts_t.shape[0], verts_t.shape[1], 3),
                    device=device,
                    dtype=verts_t.dtype,
                )
                verts_t = torch.cat([verts_t, pad], dim=0)
            else:
                verts_t = verts_t[:num_persons]
            verts_list.append(verts_t)
        except Exception as e:
            print(f"[DEBUG] Could not load GT SMPL verts for {fp}: {e}")
            return None

    return torch.stack(verts_list, dim=0) if verts_list else None


def preload_gt_smpl_joints_world(
    frame_paths: List[str | Path],
    smpl_dir: Path,
    num_persons: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load GT SMPL joints from npz in world space for the provided frames.
    Returns a tensor [F, P, 24, 3]
    """
    smpl_joints = []
    for fp in frame_paths:
        per_person = []
        for pid in range(num_persons):
            joints_w = load_gt_smpl_joints_world(smpl_dir, fp, pid, device)
            if joints_w is None:
                joints_w = torch.zeros(24, 3, device=device)
            per_person.append(joints_w)
        smpl_joints.append(torch.stack(per_person, dim=0))

    return torch.stack(smpl_joints, dim=0)


def load_smplx_joints_camera_space(
    smplx_model,
    smplx_params,
    num_persons,
    n_frames,
    c2w_proj,
    device,
):

    smplx_result = []
    for fid in range(n_frames):
        smplx_per_frame = []
        for pid in range(num_persons):
            joints_cam = smplx_joints_in_camera(
                smplx_model, smplx_params, pid, fid, c2w_proj, device
            ) # [55, 3]
            smplx_per_frame.append(joints_cam)
        smplx_result.append(torch.stack(smplx_per_frame, dim=0)) 

    return torch.stack(smplx_result, dim=0)


def preload_pred_smplx_joints_world(
    smplx_model,
    smplx_params,
    num_persons,
    n_frames,
    device,
    use_regressed: bool = False,
):
    """
    Convenience loader to get posed SMPL-X joints in model/world frame (no camera applied).
    Returns [F, P, 55, 3] and, if use_regressed=True, also [F, P, 24, 3].
    """
    smplx_result = []
    smpl_result = []
    eye_c2w = torch.eye(4, device=device, dtype=torch.float32)
    for fid in range(n_frames):
        smplx_per_person = []
        smpl_per_person = []
        for pid in range(num_persons):
            joints_world = smplx_joints_in_camera(
                smplx_model, smplx_params, pid, fid, eye_c2w, device
            )
            smplx_per_person.append(joints_world)
            if use_regressed:
                verts_world = smplx_base_vertices_in_camera(
                    smplx_model, smplx_params, pid, fid, eye_c2w, device
                )
                smpl_world = smplx_to_smpl(
                    joints_world, smplx_vertices=verts_world.unsqueeze(0) if verts_world is not None else None
                )
                if smpl_world.dim() > 2 and smpl_world.shape[0] == 1:
                    smpl_world = smpl_world.squeeze(0)
                smpl_per_person.append(smpl_world)
        smplx_result.append(torch.stack(smplx_per_person, dim=0))
        if use_regressed:
            smpl_result.append(torch.stack(smpl_per_person, dim=0))

    if use_regressed:
        return torch.stack(smplx_result, dim=0), torch.stack(smpl_result, dim=0)
    return torch.stack(smplx_result, dim=0)


def compute_s_w2m(
    smplx_model,
    smplx_params_pred,
    frame_paths,
    gt_smplx_params,
    num_persons,
    device,
    use_sim,
):
    """
    Compute similarity transform using SMPL-X joints (55) in world frame.
    """
    if use_sim and gt_smplx_params is not None:
        gt_joints_world = preload_pred_smplx_joints_world(
            smplx_model, gt_smplx_params, num_persons, len(frame_paths), device
        )  # [F,P,55,3]
        pred_joints_world = preload_pred_smplx_joints_world(
            smplx_model, smplx_params_pred, num_persons, len(frame_paths), device
        )  # [F,P,55,3]

        S_M2W_np, s, R_np, t_np, err = compute_S_M2W_from_joints(
            pred_joints_world,  # (F,P,55,3)
            gt_joints_world,    # (F,P,55,3)
        )

        S_M2W = torch.from_numpy(S_M2W_np).to(device=device, dtype=torch.float32)
        S_W2M = torch.linalg.inv(S_M2W)
    else:
        S_W2M = torch.eye(4, device=device, dtype=torch.float32)

    return S_W2M


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
#            print(f"[DEBUG] Loaded GT SMPL-X params:")
        #for k, v in smplx.items():
            #print(f"{k}: {v.shape}")
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
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)
        camera_params_path: Path = root_gt_dir_path / "cameras" / "rgb_cameras.npz"
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)
        num_tracks, num_frames = self.cfg.num_persons, self.cfg.batch_size
        render_bg_colors_template = torch.zeros(
            (num_tracks, num_frames, 3), device=self.tuner_device, dtype=torch.float32
        )  # black
        alpha_candidates = torch.tensor(
            [-0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0], device=self.tuner_device
        )
        cam_alpha: dict[int, float] = {}
        cam_alpha_stats: dict[int, dict[str, torch.Tensor | int]] = {}
        calibrate_frames = 3
        max_shift_m = 0.2

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
            cam_alpha_stats.setdefault(
                tgt_cam_id,
                {"sum_iou": torch.zeros_like(alpha_candidates), "count": 0},
            )

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
                    smplx_params = self._load_gt_smplx_params(
                        frame_paths, root_gt_dir_path / "smplx"
                    )

                    # Render with the model
                    res = self.model.animation_infer_custom(
                        self.gs_model_list,
                        self.query_points,
                        smplx_params,
                        render_c2ws=render_c2ws,
                        render_intrs=render_intrs,
                        render_bg_colors=render_bg_colors,
                        render_hw=(gt_h, gt_w),
                    )

                    delta_world = self._estimate_world_translation_offset(
                        render_mask=res["comp_mask"],
                        gt_mask=masks,
                        intr=tgt_intr,
                        c2w=tgt_c2w,
                        w2c=tgt_w2c,
                        smplx_trans=smplx_params["trans"],
                    )
                    if delta_world is not None:
                        delta_world = delta_world.clamp(-max_shift_m, max_shift_m)
                        stats = cam_alpha_stats[tgt_cam_id]
                        cached_alpha = cam_alpha.get(tgt_cam_id, None)

                        if cached_alpha is None and stats["count"] < calibrate_frames:
                            best_res = res
                            best_iou = -1.0
                            for idx, alpha in enumerate(alpha_candidates):
                                if alpha.abs() < 1e-6:
                                    cand_res = res
                                else:
                                    cand_c2ws = render_c2ws.clone()
                                    cand_c2ws[:, :, :3, 3] -= alpha * delta_world.unsqueeze(0)
                                    cand_res = self.model.animation_infer_custom(
                                        self.gs_model_list,
                                        self.query_points,
                                        smplx_params,
                                        render_c2ws=cand_c2ws,
                                        render_intrs=render_intrs,
                                        render_bg_colors=render_bg_colors,
                                        render_hw=(gt_h, gt_w),
                                    )
                                iou = self._mask_iou(cand_res["comp_mask"], masks).mean()
                                stats["sum_iou"][idx] += iou
                                if iou > best_iou:
                                    best_iou = iou
                                    best_res = cand_res
                            stats["count"] += 1
                            res = best_res
                            if stats["count"] >= calibrate_frames:
                                best_idx = int(torch.argmax(stats["sum_iou"]).item())
                                cam_alpha[tgt_cam_id] = float(alpha_candidates[best_idx].item())
                        else:
                            if cached_alpha is None:
                                best_idx = int(torch.argmax(stats["sum_iou"]).item())
                                cached_alpha = float(alpha_candidates[best_idx].item())
                                cam_alpha[tgt_cam_id] = cached_alpha
                            adj_c2ws = render_c2ws.clone()
                            adj_c2ws[:, :, :3, 3] -= cached_alpha * delta_world.unsqueeze(0)
                            res = self.model.animation_infer_custom(
                                self.gs_model_list,
                                self.query_points,
                                smplx_params,
                                render_c2ws=adj_c2ws,
                                render_intrs=render_intrs,
                                render_bg_colors=render_bg_colors,
                                render_hw=(gt_h, gt_w),
                            )

                    comp_rgb = res["comp_rgb"]  # [B, H, W, 3]

                    # Apply masks to the rendered images and gt 
                    masks3 = masks
                    if masks3.shape[-1] == 1:
                        masks3 = masks3.repeat(1, 1, 1, 3)
                    masked_render = comp_rgb # * masks3
                    masked_gt = frames # * masks3

                    # Combine masked render and masked GT with overlay
                    overlay = 0.5 * (masked_render + masked_gt)
                    columns = [masked_render, masked_gt, overlay]
                    joined = torch.cat(columns, dim=2)  # side-by-side along width
                    for i in range(joined.shape[0]):
                        save_path = save_dir / Path(frame_paths[i]).name
                        save_image(joined[i].permute(2, 0, 1), str(save_path))

        quit()

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
