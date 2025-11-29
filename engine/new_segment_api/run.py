import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM 3 text-prompted video segmentation and export masks."
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Path to a JPEG folder or an MP4 file to segment.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text prompt describing the target object(s).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where binary masks will be saved.",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index on which to place the text prompt (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the exported masked videos (default: 10).",
    )
    return parser.parse_args()


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def load_video_frames(resource_path: str):
    """
    Load all frames for masking. For large videos this will hold frames in memory.
    """
    path = Path(resource_path)
    frames = []

    if path.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {resource_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(path.glob(ext))

        if not img_paths:
            raise RuntimeError(f"No image frames found in {resource_path}")

        try:
            img_paths.sort(key=lambda p: int(p.stem))
        except Exception:
            img_paths.sort()

        for img_path in img_paths:
            frames.append(np.array(Image.open(img_path).convert("RGB")))

    return frames


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Keep only masked pixels from the frame; everything else becomes black.
    """
    frame_uint8 = _to_uint8(frame)
    mask_bool = mask.astype(bool)
    masked_frame = np.zeros_like(frame_uint8)
    masked_frame[mask_bool] = frame_uint8[mask_bool]
    return masked_frame


def save_masks(outputs_per_frame, frames, output_dir: Path):
    """
    Save a binary mask per object track in its own subdirectory.
    Also save the union of all masks. Duplicate structure under "masked_images"
    with masked RGB outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masked_dir = output_dir / "masked_images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    union_masks_dir = masks_dir / "union"
    union_masked_dir = masked_dir / "union"
    union_masks_dir.mkdir(parents=True, exist_ok=True)
    union_masked_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, masks_for_frame in outputs_per_frame.items():
        if not masks_for_frame:
            continue

        frame = frames[frame_idx]
        frame_h, frame_w = frame.shape[:2]
        combined_mask = None
        for obj_id, mask in masks_for_frame.items():
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)

            if mask_np.shape != (frame_h, frame_w):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            obj_dir = masks_dir / f"{obj_id:02d}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            binary = (mask_np > 0).astype(np.uint8)
            Image.fromarray(binary * 255).save(obj_dir / f"{frame_idx:04d}.png")

            obj_masked_dir = masked_dir / f"{obj_id:02d}"
            obj_masked_dir.mkdir(parents=True, exist_ok=True)
            masked_frame = _apply_mask(frame, binary)
            Image.fromarray(masked_frame).save(obj_masked_dir / f"{frame_idx:04d}.png")

            if combined_mask is None:
                combined_mask = np.zeros_like(binary, dtype=np.uint8)
            combined_mask |= binary

        if combined_mask is not None:
            Image.fromarray(combined_mask * 255).save(
                union_masks_dir / f"{frame_idx:04d}.png"
            )
            union_masked = _apply_mask(frame, combined_mask)
            Image.fromarray(union_masked).save(
                union_masked_dir / f"{frame_idx:04d}.png"
            )


def _sorted_frame_paths(frames_dir: Path):
    frame_paths = list(frames_dir.glob("*.png"))
    try:
        frame_paths.sort(key=lambda p: int(p.stem))
    except Exception:
        frame_paths.sort()
    return frame_paths


def export_video_from_frames(frames_dir: Path, out_path: Path, fps: int):
    frame_paths = _sorted_frame_paths(frames_dir)
    if not frame_paths:
        print(f"Warning: no frames found in {frames_dir}, skipping video export.")
        return

    try:
        start_number = int(frame_paths[0].stem)
    except ValueError:
        start_number = 0

    pattern = frames_dir / "%04d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed for {frames_dir}: {result.stderr}")


def export_videos(masked_dir: Path, fps: int):
    """
    For each track (including union) under masked_images, create a video from frames.
    """
    for subdir in masked_dir.iterdir():
        if not subdir.is_dir():
            continue
        out_path = subdir / "video.mp4"
        export_video_from_frames(subdir, out_path, fps)


def main():
    args = parse_args()

    resource_path = str(Path(args.frames).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    prompt_frame = max(0, args.prompt_frame)

    # Load frames for later masking/overlay creation.
    frames = load_video_frames(resource_path)

    # Use all available GPUs; adjust if you only want a subset.
    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    session_id = None
    try:
        # Start a session for this video / frame folder.
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": resource_path}
        )
        session_id = response["session_id"]

        # Add the text prompt on the chosen frame.
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": args.text,
            }
        )

        # Propagate segmentation across the full video.
        outputs_per_frame = propagate_in_video(predictor, session_id)

        # Flatten to {frame_idx: {obj_id: mask}} for easy saving.
        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

        save_masks(outputs_per_frame, frames, output_dir)
        export_videos(output_dir / "masked_images", fps=args.fps)
        print(f"Saved masks to {output_dir}")
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    request={"type": "close_session", "session_id": session_id}
                )
            except Exception:
                pass
        predictor.shutdown()


if __name__ == "__main__":
    main()
