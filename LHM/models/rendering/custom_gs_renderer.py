import math
from collections import defaultdict

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply


from LHM.models.rendering.custom_smplx_voxel_dense_sampling import SMPLXVoxelMeshModel
from LHM.models.rendering.utils.sh_utils import RGB2SH, SH2RGB
from LHM.models.rendering.utils.typing import *
from LHM.models.rendering.utils.utils import MLP, trunc_exp
from LHM.models.utils import LinerParameterTuner, StaticParameterTuner
from LHM.outputs.output import GaussianAppOutput


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

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb
        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)


class GSLayer(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
    ):
        super().__init__()
        self.setup_functions()

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
            clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": None,
            "rotation": None,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def constrain_forward(self, ret, constrain_dict):

        # body scaling constrain
        # gs_attr.scaling[is_constrain_body] = gs_attr.scaling[is_constrain_body].clamp(max=0.02)  # magic number, which is used to constrain 
        # hand opacity constrain 

        # force the hand's opacity to be 0.95
        # gs_attr.opacity[is_hand] = gs_attr.opacity[is_hand].clamp(min=0.95)

        # body scaling constrain
        # is_constrain_body = constrain_dict['is_constrain_body']
        is_upper_body = constrain_dict['is_upper_body']
        scaling = ret['scaling'] 
        # scaling[is_constrain_body] body_constrain= scaling[is_constrain_body].clamp(max = 0.02)
        scaling[is_upper_body] = scaling[is_upper_body].clamp(max = 0.02)
        # scaling = scaling.clamp(max=0.02)
        ret['scaling'] = scaling

        return ret

    def forward(self, x, pts, x_fine=None, constrain_dict=None):
        assert len(x.shape) == 2
        ret = {}
        for k in self.attr_dict:
            layer = self.out_layers[k]

            v = layer(x)
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )  # constant rotation
                else:
                    # v = torch.nn.functional.normalize(v)
                    v = self.rotation_activation(v)
            elif k == "scaling":
                # v = trunc_exp(v)
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
                    v = self.rgb_activation(v)

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)

        return GaussianAppOutput(**ret)


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
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.smpl_type = smpl_type
        assert self.smpl_type in ["smplx", "smplx_0", "smplx_1", "smplx_2"]

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree


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

        self.mlp_network_config = mlp_network_config

        # using to mapping transformer decode feature to regression features. as decode feature is processed by NormLayer.
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

        self.gs_net = GSLayer(
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            init_scaling=-5.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            use_fine_feat=(
                True
                if decode_with_extra_info is not None
                and decode_with_extra_info["type"] is not None
                else False
            ),
        )

    def hyper_step(self, step):
        self.gs_net.hyper_step(step)

    def forward_single_view(
        self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
        }

        return ret

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
            # Debug extent of canonical points (before LBS)

#             assert mean_3d.shape[1] == 3 and len(mean_3d.shape) == 2, "Unexpected shape for mean 3d"
            # x_min, x_max = mean_3d[:, 0].min(), mean_3d[:, 0].max()
            # y_min, y_max = mean_3d[:, 1].min(), mean_3d[:, 1].max()
            # z_min, z_max = mean_3d[:, 2].min(), mean_3d[:, 2].max()
            # print(f"[DEBUG] Canonical GS extent along X-axis min: {x_min.detach().cpu().numpy()}, max: {x_max.detach().cpu().numpy()}")
            # print(f"[DEBUG] Canonical GS extent along Y-axis min: {y_min.detach().cpu().numpy()}, max: {y_max.detach().cpu().numpy()}")
            # print(f"[DEBUG] Canonical GS extent along Z-axis min: {z_min.detach().cpu().numpy()}, max: {z_max.detach().cpu().numpy()}")
            # quit()

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
            mean_3d, transform_matrix = (
                self.smplx_model.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    merge_smplx_data,
                    query_points,
                    transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    device=device,
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
                use_rgb=self.gs_net.use_rgb,
            )  # [N, 3]

            if i == num_view - 1:
                # print(f"  Appending to canonical GS list.")
                cano_gs_list.append(gs_copy)
            else:
                # print(f"  Appending to GS list.")
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list


    def forward_gs_attr(self, x, query_points, smplx_data, debug=False, x_fine=None):
        """
        x: [N, C] Float[Tensor, "Np Cp"],
        query_points: [N, 3] Float[Tensor, "Np 3"]
        """
        device = x.device
        if self.mlp_network_config is not None:
            # x is processed by LayerNorm
            x = self.mlp_net(x)
            if x_fine is not None:
                x_fine = self.mlp_net(x_fine)

        # NOTE that gs_attr contains offset xyz
        is_constrain_body = self.smplx_model.is_constrain_body
        is_hands =  self.smplx_model.is_rhand + self.smplx_model.is_lhand 
        is_upper_body = self.smplx_model.is_upper_body

        constrain_dict=dict(
            is_constrain_body=is_constrain_body,
            is_hands=is_hands,
            is_upper_body=is_upper_body,
        )

        gs_attr: GaussianAppOutput = self.gs_net(x, query_points, x_fine, constrain_dict)

        return gs_attr

    def get_query_points(self, smplx_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                # print(smplx_data["betas"].shape, smplx_data["face_offset"].shape, smplx_data["joint_offset"].shape)
                positions, pos_wo_upsample, transform_mat_neutral_pose = (
                    self.smplx_model.get_query_points(smplx_data, device=device)
                )  # [B, N, 3]

        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 55, 4, 4]
        )
        return positions, smplx_data


    def query_latent_feat(
        self,
        positions: Float[Tensor, "*B N1 3"],
        smplx_data,
        latent_feat: Float[Tensor, "*B N2 C"],
        extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, smplx_data = self.get_query_points(smplx_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)

            gs_feats = self.decoder_cross_attn(
                pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
            )

        return gs_feats, positions, smplx_data


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
                self.forward_single_view(
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

        # debug = True
        if debug:
            import cv2

            cv2.imwrite(
                "fuck.png",
                (out["comp_rgb"].detach().cpu().numpy()[0, ..., ::-1] * 255).astype(
                    np.uint8
                ),
            )

        return out


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

    def forward_gs(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs,
    ):

        batch_size = gs_hidden_features.shape[0]

        # obtain gs_features embedding, cur points position, and also smplx params
        query_gs_features, query_points, smplx_data = self.query_latent_feat(
            query_points, smplx_data, gs_hidden_features, additional_features
        )

        gs_attr_list = []
        for b in range(batch_size):
            if isinstance(query_gs_features, dict):
                gs_attr = self.forward_gs_attr(
                    query_gs_features["coarse"][b],
                    query_points[b],
                    None,
                    debug,
                    x_fine=query_gs_features["fine"][b],
                )
            else:
                gs_attr = self.forward_gs_attr(
                    query_gs_features[b], query_points[b], None, debug
                )
            gs_attr_list.append(gs_attr)

        return gs_attr_list, query_points, smplx_data


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
        cano_out_list = []  # inference DO NOT use

        N_view = smplx_data["root_pose"].shape[1]
        # print(f"[DEBUG] N_view: {N_view}")


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
                use_rgb=self.gs_net.use_rgb,
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

    def forward(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        **kwargs,
    ):

        # need shape_params of smplx_data to get querty points and get "transform_mat_neutral_pose"
        # only forward gs params
        gs_attr_list, query_points, smplx_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            smplx_data=smplx_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            smplx_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs["df_data"],
        )
        out["gs_attr"] = gs_attr_list

        return out
