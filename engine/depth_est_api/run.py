import os
import argparse
from pathlib import Path

# Ensure model weights cache to desired location
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from depth_anything_3.api import DepthAnything3
import subprocess


def load_frames(frames_dir: Path):
    frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg")) + sorted(
        frames_dir.glob("*.jpeg")
    )
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    return frame_paths, images


def save_depth_maps(depth_batch: torch.Tensor, frame_paths, output_dir: Path, masks_dir: Path):
    raw_dir = output_dir / "raw"
    png_dir = output_dir / "png"
    masked_dir = output_dir / "masked_png"
    raw_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(depth_batch, torch.Tensor):
        depth_np = depth_batch.cpu().numpy()
    else:
        depth_np = np.asarray(depth_batch)
    for dp, src_path in zip(depth_np, frame_paths):
        # Upsample depth to original image resolution
        orig_hw = Image.open(src_path).size[::-1]  # (H,W)
        dp_resized = np.array(Image.fromarray(dp).resize(orig_hw[::-1], resample=Image.BILINEAR))

        np.save(raw_dir / (src_path.stem + ".npy"), dp.astype(np.float32))
        # Normalize depth for visualization; handle zero/inf safely.
        depth_vis = dp_resized.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0.0
        # Use top 95% range (offset from max) to emphasize closer depths
        dmin, dmax = np.percentile(depth_vis, [5, 100])
        if dmax > dmin:
            depth_norm = (depth_vis - dmin) / (dmax - dmin)
        else:
            depth_norm = depth_vis * 0.0
        # Apply colormap for nicer visualization
        depth_color = (cm.magma(depth_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(depth_color).save(png_dir / (src_path.stem + ".png"))

        # Masked depth visualization
        mask_path = masks_dir / src_path.name
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize(orig_hw[::-1], resample=Image.NEAREST)
            mask_arr = np.array(mask).astype(np.float32) / 255.0
            mask_arr = (mask_arr > 0.5).astype(np.float32)
            depth_masked = depth_vis * mask_arr
            # Normalize using only masked pixels
            masked_vals = depth_masked[mask_arr > 0]
            if masked_vals.size > 0:
                dmin_m, dmax_m = masked_vals.min(), masked_vals.max()
                if dmax_m > dmin_m:
                    depth_masked_norm = (depth_masked - dmin_m) / (dmax_m - dmin_m)
                else:
                    depth_masked_norm = depth_masked * 0.0
            else:
                depth_masked_norm = depth_masked * 0.0
            depth_masked_norm = np.clip(depth_masked_norm, 0.0, 1.0)
            depth_masked_color = (cm.magma(depth_masked_norm)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(depth_masked_color).save(masked_dir / (src_path.stem + ".png"))

    # Build a video from the PNGs for quick inspection.
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(png_dir / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(png_dir / "depth_vis.mp4"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARN] ffmpeg video export failed: {e}")

    # Build a video for masked depth
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-start_number",
            "0",
            "-i",
            str(masked_dir / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(masked_dir / "depth_masked_vis.mp4"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARN] ffmpeg masked video export failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 metric model over frames and save depth maps."
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Base output directory containing frames/")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    depth_out_dir = output_dir / "depth_maps"
    masks_dir = output_dir / "masks" / "union"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").to(device)

    frame_paths, images = load_frames(frames_dir)
    # DepthAnything3 expects a non-None export_format; use "npz" but ignore exported files.
    prediction = model.inference(
        images,
        export_dir=str(depth_out_dir / "raw"),
        export_format="npz",
    )

    # Depth is canonical metric depth (meters). No focal scaling needed for meters here.
    save_depth_maps(prediction.depth, frame_paths, depth_out_dir, masks_dir)
    print(f"Saved {len(frame_paths)} depth maps to {depth_out_dir}")


if __name__ == "__main__":
    main()
