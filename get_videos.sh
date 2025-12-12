#!/bin/bash
set -euo pipefail
module load gcc ffmpeg


cam_id=52
exp_name="gsplat_depth_loss_w1.0_tune_color"

EPOCH0="/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance/lhm/evaluation/$exp_name/epoch_0000/$cam_id/cam_${cam_id}_nvs_epoch_0000.mp4"
EPOCH25="/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance/lhm/evaluation/$exp_name/epoch_0025/$cam_id/cam_${cam_id}_nvs_epoch_0025.mp4"
OUTPUT_DIR="/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance/lhm/evaluation/$exp_name"
OUTPUT_MP4="$OUTPUT_DIR/cam_${cam_id}_nvs_epoch_0_25_gt.mp4"
OUTPUT_GIF="$OUTPUT_DIR/cam_${cam_id}_nvs_epoch_0_25_gt.gif"
mkdir -p "$OUTPUT_DIR"

# temporary files
tmp0=$(mktemp --suffix=.mp4)
tmp25=$(mktemp --suffix=.mp4)
tmpgt=$(mktemp --suffix=.mp4)
tmpstack=$(mktemp --suffix=.mp4)

tmp_palette=$(mktemp --suffix=.png)

tmpgif_in=$(mktemp --suffix=.mp4)

cleanup() {
  rm -f "$tmp0" "$tmp25" "$tmpgt" "$tmpstack" "$tmp_palette" "$tmpgif_in"
}
trap cleanup EXIT

# Crop left half (prediction) from epoch 0
ffmpeg -y -i "$EPOCH0" -filter_complex "crop=iw/2:ih:0:0" "$tmp0"
# Crop left half (prediction) from epoch 25
ffmpeg -y -i "$EPOCH25" -filter_complex "crop=iw/2:ih:0:0" "$tmp25"
# Crop right half (ground truth) from epoch 0 (assumed same GT)
ffmpeg -y -i "$EPOCH0" -filter_complex "crop=iw/2:ih:iw/2:0" "$tmpgt"

# Stack videos horizontally: [pred0 | pred25 | gt]
ffmpeg -y \
  -i "$tmp0" \
  -i "$tmp25" \
  -i "$tmpgt" \
  -filter_complex "[0:v][1:v][2:v]hstack=inputs=3" \
  "$tmpstack"

# Copy stacked mp4 to final location
cp "$tmpstack" "$OUTPUT_MP4"

# Create GIF (palette for quality)
ffmpeg -y -i "$tmpstack" -vf "fps=10,scale=960:-1:flags=lanczos,palettegen" "$tmp_palette"
ffmpeg -y -i "$tmpstack" -i "$tmp_palette" -filter_complex "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUTPUT_GIF"

echo "Videos written to $OUTPUT_MP4 and $OUTPUT_GIF"
