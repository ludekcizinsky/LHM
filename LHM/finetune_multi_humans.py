"""
Usage:
    python LHM/infer_multi_humans.py --gs_model_dir /path/to/gs

Example:
    python LHM/infer_multi_humans.py --gs_model_dir=/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/lhm/inference_results
"""


import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser
from pathlib import Path

from LHM.models import ModelHumanLRMSapdinoBodyHeadSD3_5
from LHM.utils.hf_hub import wrap_model_hub
from LHM.runners.infer.base_inferrer import Inferrer

import torch
import numpy as np

from LHM.outputs.output import GaussianAppOutput

from dataclasses import fields

def enable_gaussian_grads(gauss: GaussianAppOutput, *, detach_to_leaf=False):
    for f in fields(gauss):
        v = getattr(gauss, f.name)
        if torch.is_tensor(v):
            if detach_to_leaf:
                v = v.detach().requires_grad_()
                setattr(gauss, f.name, v)
            else:
                v.requires_grad_(True)


class MultiHumanFinetuner(Inferrer):
    EXP_TYPE = "multi_human_finetune"

    def __init__(self, output_dir: Path, render_save_dir: Path, scene_name: str):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_gs_model(output_dir)
        self.model : ModelHumanLRMSapdinoBodyHeadSD3_5 = self._build_model().to(device)
        self.render_save_dir = render_save_dir
        self.render_save_dir.mkdir(parents=True, exist_ok=True)
        self.scene_name = scene_name

    def _build_model(self):
        hf_model_cls = wrap_model_hub(ModelHumanLRMSapdinoBodyHeadSD3_5)
        model_name = "/scratch/izar/cizinsky/pretrained/huggingface/models--3DAIGC--LHM-1B/snapshots/cd8a1cc900a557d83187cfc2e0a91cef3eba969d/"
        model = hf_model_cls.from_pretrained(model_name)
        return model

    def load_gs_model(self, root_output_dir: Path):
        root_gs_model_dir = root_output_dir / "initial_scene_recon"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track_ids = os.listdir(root_gs_model_dir)
        track_ids = sorted([track_id for track_id in track_ids if (root_gs_model_dir / track_id).is_dir()])
        print(f"[DEBUG] Found {len(track_ids)} humans for inference: {track_ids}")
        self.all_model_list = []
        for track_id in track_ids:
            gs_model_dir = root_gs_model_dir / track_id
            gs_model_list = torch.load(gs_model_dir / "gs_model_list.pt", map_location=device)
            query_points = torch.load(gs_model_dir / "query_points.pt", map_location=device)
            transform_mat_neutral_pose = torch.load(gs_model_dir / "transform_mat_neutral_pose.pt", map_location=device)
            motion_seq = torch.load(gs_model_dir / "motion_seq.pt", map_location=device)
            shape_params = torch.from_numpy(np.load(gs_model_dir / "shape_params.npy")).unsqueeze(0).to(device)
            model = (gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_params)
            self.all_model_list.append(model)
        print(f"[DEBUG] Loaded GS models for {len(self.all_model_list)} humans.")

    def infer_single(self, *args, **kwargs):
        pass

    def infer(self):
        pass

    def _get_joined_inference_inputs(self):

        gs_model_list = []
        query_points = None
        transform_mat_neutral_pose = None
        motion_seq = None
        shape_param = None
        for track_idx in range(len(self.all_model_list)):
            p_gs_model_list, p_query_points, p_transform_mat_neutral_pose, p_motion_seq, p_shape_param = self.all_model_list[track_idx]
            enable_gaussian_grads(p_gs_model_list[0], detach_to_leaf=True)
            gs_model_list.extend(p_gs_model_list)
            if query_points is None:
                query_points = p_query_points
            else:
                query_points = torch.cat([query_points, p_query_points], dim=0)
 
            if transform_mat_neutral_pose is None:
                transform_mat_neutral_pose = p_transform_mat_neutral_pose
            else:
                transform_mat_neutral_pose = torch.cat([transform_mat_neutral_pose, p_transform_mat_neutral_pose], dim=0)
            
            if motion_seq is None:
                motion_seq = p_motion_seq
            else:
                for key in motion_seq["smplx_params"].keys():
                    motion_seq["smplx_params"][key] = torch.cat([motion_seq["smplx_params"][key], p_motion_seq["smplx_params"][key]], dim=0)
            
            if shape_param is None:
                shape_param = p_shape_param
            else:
                shape_param = torch.cat([shape_param, p_shape_param], dim=0)

        print(f"[DEBUG] len of gs model list: {len(gs_model_list)}")
        print(f"[DEBUG] shape of query points: {query_points.shape}")
        print(f"[DEBUG] shape of transform_mat_neutral_pose: {transform_mat_neutral_pose.shape}")
        print(f"[DEBUG] shape of shape param: {shape_param.shape}")
        for k, v in motion_seq["smplx_params"].items():
            print(f"[DEBUG] motion_seq smplx_params key:{k}, shape:{v.shape}")

        return gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_param

    def finetune(self):
        # Prepare inputs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_param = self._get_joined_inference_inputs()

        batch_size = 40  # avoid memeory out!

        camera_size = len(motion_seq["motion_seqs"])
        for batch_i in range(0, camera_size, batch_size):
            print(f"[DEBUG] batch: {batch_i}, total: {camera_size //batch_size +1} ")

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


            batch_smplx_params = dict()
            batch_smplx_params["betas"] = shape_param.to(device)
            batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
            for key in keys:
                batch_smplx_params[key] = motion_seq["smplx_params"][key][
                    :, batch_i : batch_i + batch_size
                ].to(device)

            res = self.model.animation_infer_custom(gs_model_list, query_points, batch_smplx_params,
                render_c2ws=motion_seq["render_c2ws"][
                    :, batch_i : batch_i + batch_size
                ].to(device),
                render_intrs=motion_seq["render_intrs"][
                    :, batch_i : batch_i + batch_size
                ].to(device),
                render_bg_colors=motion_seq["render_bg_colors"][
                    :, batch_i : batch_i + batch_size
                ].to(device),
                )

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0
            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            
            # assert batch rgb to have grad
            assert batch_rgb.requires_grad, "[ERROR] batch_rgb does not have grad!"

            del res
            torch.cuda.empty_cache()
        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--render_save_dir", type=Path)
    parser.add_argument("--scene_name", type=str)
    args = parser.parse_args()

    tuner = MultiHumanFinetuner(output_dir=args.output_dir, render_save_dir=args.render_save_dir, scene_name=args.scene_name)
    tuner.finetune()
