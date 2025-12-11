import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply


from LHM.models.rendering.custom_smplx_voxel_dense_sampling import SMPLXVoxelMeshModel
from LHM.models.rendering.utils.typing import *
from LHM.models.rendering.utils.utils import MLP
from LHM.outputs.output import GaussianAppOutput

from gsplat.rendering import rasterization


def auto_repeat_size(tensor, repeat_num, axis=0):
    repeat_size = [1] * tensor.dim()
    repeat_size[axis] = repeat_num
    return repeat_size


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y



class Camera:
    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(w2c.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )
        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
        )


class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # rgb activation function
        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, use_rgb=False) -> None:
        """
        Initializes the GSRenderer object.
        Args:
            xyz (Tensor): The xyz coordinates.
            opacity (Tensor): The opacity values.
            rotation (Tensor): The rotation values.
            scaling (Tensor): The scaling values.
            before_activate: if True, the output appearance is needed to process by activation function.
            shs (Tensor): The spherical harmonics coefficients.
            use_rgb (bool, optional): Indicates whether shs represents RGB values. Defaults to False.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]

        self.use_rgb = use_rgb  # shs indicates rgb?
 
class GS3DRenderer(nn.Module):
    def __init__(
        self,
        human_model_path,
        subdivide_num,
        smpl_type,
        feat_dim,
        query_dim,
        use_rgb,
        sh_degree,
        xyz_offset_max_step,
        mlp_network_config,
        expr_param_dim,
        shape_param_dim,
        clip_scaling=0.2,
        cano_pose_type=0,
        decoder_mlp=False,
        skip_decoder=False,
        fix_opacity=False,
        fix_rotation=False,
        decode_with_extra_info=None,
        gradient_checkpointing=False,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,  # only use for dense_smaple_smplx
    ):

        super().__init__()
        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        self.use_rgb = use_rgb
        self.smplx_model = SMPLXVoxelMeshModel(
            human_model_path,
            gender="neutral",
            subdivide_num=subdivide_num,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            cano_pose_type=cano_pose_type,
            dense_sample_points=dense_sample_pts,
            apply_pose_blendshape=apply_pose_blendshape,
        )


    # --------------------------------------------------------------------------
    # Animate + Render
    # --------------------------------------------------------------------------
    def get_query_points(self, smplx_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                positions, pos_wo_upsample, transform_mat_neutral_pose = (
                    self.smplx_model.get_query_points(smplx_data, device=device)
                )  # [B, N, 3]

        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 55, 4, 4]
        )
        return positions, smplx_data

    def forward_single_view_gsplat(
        self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
    ):

        # Prepare inputs for rasterization
        means = gs.xyz
        quats = gs.rotation
        scales = gs.scaling
        opacities = gs.opacity.squeeze(-1)
        colors = gs.shs.squeeze(1).float()
        viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)
        Ks = viewpoint_camera.intrinsic[:3, :3].unsqueeze(0)
        width = viewpoint_camera.width
        height = viewpoint_camera.height
        near_plane = viewpoint_camera.znear
        far_plane = viewpoint_camera.zfar
        render_mode = "RGB+D"

        # Perform rasterization
        renders, alphas, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            packed=False,
        )


        # Pack outputs
        ret = {
            "comp_rgb": renders.squeeze(0)[..., :3],  # [H, W, 3]
            "comp_rgb_bg": background_color,
            "comp_mask": alphas.squeeze(0),  # [H, W, 1]
            "comp_depth": renders.squeeze(0)[..., 3:],  # [H, W, 1]
        }

        return ret

    def forward_single_batch_custom(
        self,
        gs_list: list[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
        debug: bool = False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            out_list.append(
                self.forward_single_view_gsplat(
                    gs_list[v_idx],
                    Camera.from_c2w(c2w, intrinsic, height, width),
                    background_color[v_idx],
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list

        return out

    def animate_gs_model_custom(
        self, gs_attr: GaussianAppOutput, query_points, smplx_data
    ):
        """
        query_points: [N, 3]
        """

        device = gs_attr.offset_xyz.device

        # build cano_dependent_pose
        cano_smplx_data_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
        ]

        # Build a pose batch containing both the provided poses and an added canonical pose.
        merge_smplx_data = dict()
        for cano_smplx_data_key in cano_smplx_data_keys:
            # Incoming pose(s) for this key: shape [Nv, ...].
            warp_data = smplx_data[cano_smplx_data_key]
            # One canonical pose slot to append: shape [1, ...], zero-initialized.
            cano_pose = torch.zeros_like(warp_data[:1])

            if cano_smplx_data_key == "body_pose":
                # Define canonical body pose as a light A-pose (rotate shoulders).
                cano_pose[0, 15, -1] = -math.pi / 6
                cano_pose[0, 16, -1] = +math.pi / 6

            # Stack the posed input and an extra canonical pose for each key.
            merge_pose = torch.cat([warp_data, cano_pose], dim=0)
            merge_smplx_data[cano_smplx_data_key] = merge_pose

        # Copy over shape params and neutral-pose transforms unchanged.
        merge_smplx_data["betas"] = smplx_data["betas"]
        merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
            "transform_mat_neutral_pose"
        ]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # Start from canonical points offset by learned xyz.
            mean_3d = (
                query_points + gs_attr.offset_xyz
            )  # [N, 3]  # canonical space offset.

            # matrix to warp predefined pose to zero-pose
            transform_mat_neutral_pose = merge_smplx_data[
                "transform_mat_neutral_pose"
            ]  # [55, 4, 4]
            num_view = merge_smplx_data["body_pose"].shape[0]  # [Nv, 21, 3]
            # Broadcast inputs across all poses/canonical view.
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1) # [Nv, N, 3]
            transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
                num_view, 1, 1, 1
            )

            # print(mean_3d.shape, transform_mat_neutral_pose.shape, query_points.shape, smplx_data["body_pose"].shape, smplx_data["betas"].shape)
            mean_3d, transform_matrix, posed_joints = (
                self.smplx_model.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    merge_smplx_data,
                    query_points,
                    transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    device=device,
                    return_joints=True,
                )
            )  # [B, N, 3]

            # rotation appearance from canonical space to view_posed
            num_view, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[:, :, :3, :3]

            # Convert per-point rotation to quaternion and normalize.
            rigid_rotation_matrix = torch.nn.functional.normalize(
                matrix_to_quaternion(transform_rotation), dim=-1
            )
            I = matrix_to_quaternion(torch.eye(3)).to(device)

            # inference constrain
            is_constrain_body = self.smplx_model.is_constrain_body # [N,]
            # print(f"[DEBUG] Shape of is_constrain_body: {is_constrain_body.shape}")
            rigid_rotation_matrix[:, is_constrain_body] = I
            # Canonical gaussian rotations replicated per view.
            rotation_neutral_pose = gs_attr.rotation.unsqueeze(0).repeat(num_view, 1, 1)

            # QUATERNION MULTIPLY
            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )
            # rotation_pose_verts = rotation_neutral_pose

        gs_list = []
        cano_gs_list = []
        for i in range(num_view):

            gs_copy = GaussianModel(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                # rotation=gs_attr.rotation,
                rotation=rotation_pose_verts[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                use_rgb=self.use_rgb,
            )  # [N, 3]

            if i == num_view - 1:
                # print(f"  Appending to canonical GS list.")
                cano_gs_list.append(gs_copy)
            else:
                # print(f"  Appending to GS list.")
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list


    def get_single_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[
                bidx
            ]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[
                    bidx : bidx + 1
                ]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch

    def get_single_view_smpl_data(self, smpl_data, vidx):
        # print(f"[DEBUG] Getting single view smpl data for view index: {vidx}")
        smpl_data_single_view = {}
        for k, v in smpl_data.items():
            # assert v.shape[0] == 1
            if (
                k == "betas"
                or (k == "joint_offset")
                or (k == "face_offset")
                or (k == "transform_mat_neutral_pose")
            ):
                smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
                # print(f"  Key {k} shape: {smpl_data_single_view[k].shape}")
            else:
                smpl_data_single_view[k] = v[
                    :, vidx : vidx + 1
                ]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
                # print(f"  Key {k} shape: {smpl_data_single_view[k].shape}")
        return smpl_data_single_view

    def forward_animate_gs_custom(
        self,
        gs_attr_list,
        query_points,
        smplx_data,
        c2w,
        intrinsic,
        height,
        width,
        background_color,
        debug=False,
        df_data=None,  # deepfashion-style dataset
    ):
        batch_size = len(gs_attr_list)
        out_list = []

        # step 1: animate gs model = canonical -> posed view
        all_posed_gs_list = []
        for person_idx in range(batch_size):
            gs_attr = gs_attr_list[person_idx]
            query_pt = query_points[person_idx]
            posed_gs_list, _ = self.animate_gs_model_custom(
                gs_attr,
                query_pt,
                self.get_single_batch_smpl_data(smplx_data, person_idx),
            )
            posed_gs = posed_gs_list[0] 
            all_posed_gs_list.append(posed_gs)

        # merge the gs of all persons
        merged_xyz = torch.cat([gs.xyz for gs in all_posed_gs_list], dim=0)
        merged_opacity = torch.cat([gs.opacity for gs in all_posed_gs_list], dim=0)
        merged_rotation = torch.cat([gs.rotation for gs in all_posed_gs_list], dim=0)
        merged_scaling = torch.cat([gs.scaling for gs in all_posed_gs_list], dim=0)
        merged_shs = torch.cat([gs.shs for gs in all_posed_gs_list], dim=0)
        animatable_gs_model_list = [
            GaussianModel(
                xyz=merged_xyz,
                opacity=merged_opacity,
                rotation=merged_rotation,
                scaling=merged_scaling,
                shs=merged_shs,
                use_rgb=self.use_rgb,
            )
        ]

        # step 2: gs render animated gs model = posed view -> render image
        b = 0
        render_result = self.forward_single_batch_custom(
                animatable_gs_model_list,
                c2w[b],
                intrinsic[b],
                height,
                width,
                background_color[b] if background_color is not None else None,
                debug=debug,
        )
        out_list.append(render_result)

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v

        out["comp_rgb"] = out["comp_rgb"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 3, H, W]
        out["comp_mask"] = out["comp_mask"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        out["comp_depth"] = out["comp_depth"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        return out
