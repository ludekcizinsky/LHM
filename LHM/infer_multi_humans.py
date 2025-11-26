"""
Usage:
    python LHM/infer_multi_humans.py --gs_model_dir /path/to/gs

Example:
    python LHM/infer_multi_humans.py --gs_model_dir=/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/lhm/inference_results
"""


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

from LHM.utils.ffmpeg_utils import images_to_video

class MultiHumanInferrer(Inferrer):
    EXP_TYPE = "multi_human_infer"

    def __init__(self, gs_model_dir: Path, save_dir: Path):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_gs_model(gs_model_dir)
        self.model : ModelHumanLRMSapdinoBodyHeadSD3_5 = self._build_model().to(device)
        self.save_dir = save_dir

    def _build_model(self):
        hf_model_cls = wrap_model_hub(ModelHumanLRMSapdinoBodyHeadSD3_5)
        model_name = "/scratch/izar/cizinsky/pretrained/huggingface/models--3DAIGC--LHM-1B/snapshots/cd8a1cc900a557d83187cfc2e0a91cef3eba969d/"
        model = hf_model_cls.from_pretrained(model_name)
        return model

    def load_gs_model(self, root_gs_model_dir: Path):
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

    def infer(self, track_idx: int):
        # parse inputs for given track idx
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gs_model_list, query_points, transform_mat_neutral_pose, motion_seq, shape_param = self.all_model_list[track_idx]

        batch_list = [] 
        batch_size = 40  # avoid memeory out!

        camera_size = len(motion_seq["motion_seqs"])
        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():

                print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

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

                res = self.model.animation_infer(gs_model_list, query_points, batch_smplx_params,
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
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)
        dump_video_path = self.save_dir / f"human_{track_idx:02d}_inference.mp4"

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print(f"save video to {dump_video_path}")


        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=20,
            gradio_codec=False,
            verbose=True,
        )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gs_model_dir", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--track_id", type=int)
    args = parser.parse_args()

    inferrer = MultiHumanInferrer(gs_model_dir=args.gs_model_dir, save_dir=args.save_dir)
    inferrer.infer(track_idx=args.track_id)