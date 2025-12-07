import os
from typing import List, Tuple
import numpy as np
import torch
import pyrender
import trimesh

def smplx_base_vertices_in_camera(
    smplx_model,
    smplx_params: dict,
    pid: int,
    frame_idx: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute base (non-upsampled) SMPL-X vertices in camera coordinates for a given person and frame.
    Assumes the provided SMPL-X parameters are already expressed in the camera frame.
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
        # Since params are camera-relative, the layer output is already in camera coords.
        return output.vertices[0]  # [V, 3]
    except Exception as e:
        print(f"[DEBUG] Could not compute base SMPL-X verts in camera: {e}")
        return None

def overlay_smplx_mesh_pyrender(
    images: torch.Tensor,
    smplx_params: dict,
    smplx_model,
    intr: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    mesh_alpha: float = 0.7,
) -> torch.Tensor:
    """
    Render SMPL-X meshes with trimesh+pyrender and alpha-blend them over images.
    
    images: [F, H, W, 3] float in [0,1]
    smplx_params: dict with shapes [P, F, ...]
    intr: [3,3] or [4,4] intrinsics
    mesh_color: RGB in [0,1]; mesh_alpha: opacity for the mesh layer
    """

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
                    smplx_model, smplx_params, pid, fi, device
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
