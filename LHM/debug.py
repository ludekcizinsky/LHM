import os
from typing import List, Tuple
import numpy as np
import torch
import pyrender
import trimesh

def smplx_base_vertices(
    smplx_model,
    smplx_params: dict,
    pid: int,
    frame_idx: int,
    device: torch.device,
) -> torch.Tensor | None:
    layer = getattr(smplx_model, "smplx_layer", None).to(device)
    params = {
        "global_orient": smplx_params["root_pose"][pid : pid + 1, frame_idx],
        "body_pose": smplx_params["body_pose"][pid : pid + 1, frame_idx],
        "jaw_pose": smplx_params["jaw_pose"][pid : pid + 1, frame_idx],
        "leye_pose": smplx_params["leye_pose"][pid : pid + 1, frame_idx],
        "reye_pose": smplx_params["reye_pose"][pid : pid + 1, frame_idx],
        "left_hand_pose": smplx_params["lhand_pose"][pid : pid + 1, frame_idx],
        "right_hand_pose": smplx_params["rhand_pose"][pid : pid + 1, frame_idx],
        "betas": smplx_params["betas"][pid : pid + 1],
        "transl": smplx_params["trans"][pid : pid + 1, frame_idx],
        "expression": smplx_params["expr"][pid : pid + 1, frame_idx]
    }
    output = layer(**{k: v.to(device) for k, v in params.items()})
    return output.vertices[0]  # [V, 3]

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
    mesh_color: RGB in [0,1]; mesh_alpha: opacity for the mesh layer
    """

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    layer = getattr(smplx_model, "smplx_layer", None)
    faces = getattr(layer, "faces", None) if layer is not None else None
    faces_np = np.asarray(faces, dtype=np.int64)

    intr_cpu = intr.detach().cpu()
    fx, fy, cx, cy = (
        float(intr_cpu[0, 0]),
        float(intr_cpu[1, 1]),
        float(intr_cpu[0, 2]),
        float(intr_cpu[1, 2]),
    )

    num_frames = images.shape[0]
    num_people = smplx_params["betas"].shape[0]
    H, W = images.shape[1], images.shape[2]

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    cv_to_gl = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=c2w.device,
        dtype=c2w.dtype,
    )

    out_frames: List[torch.Tensor] = []
    out_ious: List[float] = []
    for fi in range(num_frames):
        base_img = (images[fi].detach().cpu().numpy() * 255).astype(np.uint8)
        depth_map = np.ones((H, W)) * np.inf
        base_img_float = base_img.astype(np.float32)
        overlay_img = base_img_float.copy()

        p_ious = list()
        for pid in range(num_people):
            verts = smplx_base_vertices(
                smplx_model, smplx_params, pid, fi, device
            )
            verts_np = verts.detach().cpu().numpy()

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
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

            scene = pyrender.Scene(
                bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5)
            )
            scene.add(mesh, "mesh")
            camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
            w2c_cv = torch.inverse(c2w)
            w2c_gl = torch.matmul(cv_to_gl, w2c_cv)
            c2w_gl = torch.inverse(w2c_gl)
            pose = c2w_gl.clone()
            pose[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
            scene.add(camera, pose=pose.detach().cpu().numpy())
            color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

            valid_mask = (rend_depth < depth_map) & (rend_depth > 0)
            depth_map[valid_mask] = rend_depth[valid_mask]
            valid_mask = valid_mask[..., None]


            color_mask = np.any(color[..., :3] != 0, axis=-1)
            overlay_mask = np.any(overlay_img[..., :3] != 0, axis=-1) 
            union_mask = color_mask | overlay_mask
            if union_mask.any():
                intersection = color_mask & overlay_mask
                iou = float(
                    intersection.sum(dtype=np.float32)
                    / union_mask.sum(dtype=np.float32)
                )
            else:
                iou = 0.0
            
            p_ious.append(iou)

            # overlay
            overlay_img = valid_mask * color[..., :3] + (1.0 - valid_mask) * overlay_img

        overlay_tensor = (
            torch.from_numpy(overlay_img).to(device=images.device, dtype=images.dtype) / 255.0
        )
        out_frames.append(overlay_tensor)
        out_ious.append(p_ious)
    renderer.delete()

    return torch.stack(out_frames, dim=0), out_ious
