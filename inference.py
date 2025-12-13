import os
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import argparse
from omegaconf import OmegaConf
from pathlib import Path

from LHM.runners.infer.human_lrm import HumanLRMInferrer
from LHM.utils.download_utils import download_from_url
from LHM.utils.model_card import MODEL_CONFIG
from LHM.utils.model_download_utils import AutoModelQuery


def parse_configs():

    if not os.path.exists('/scratch/izar/cizinsky/pretrained/dense_sample_points/1_20000.ply'):
        os.makedirs('/scratch/izar/cizinsky/pretrained/dense_sample_points/', exist_ok=True)
        download_from_url('https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply','/scratch/izar/cizinsky/pretrained/dense_sample_points/')

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str)
    parser.add_argument(
        "--save_dir",
        type=Path
    )
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)
    query_model = AutoModelQuery()

    model_name = cli_cfg.model_name
    model_path= query_model.query(model_name) 
    cli_cfg.model_name = model_path 
    model_params = model_name.split('-')[1] 
    model_config = MODEL_CONFIG[model_params] 


    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg



if __name__ == "__main__":
    cfg = parse_configs()
    inferrer = HumanLRMInferrer(cfg)
    inferrer.run()