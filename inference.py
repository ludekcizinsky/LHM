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

    # Parse cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path
    )
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # Resolve model path
    query_model = AutoModelQuery()
    model_name = cli_cfg.model_name
    model_path = query_model.query(model_name) 
    cli_cfg.model_name = model_path 
    model_params = model_name.split('-')[1] 
    model_config = MODEL_CONFIG[model_params] 

    # Load model config
    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high

    # Merge configs
    cfg.merge_with(cli_cfg)
    assert cfg.model_name is not None, "model_name is required"

    return cfg


if __name__ == "__main__":
    cfg = parse_configs()
    inferrer = HumanLRMInferrer(cfg)
    inferrer.run()