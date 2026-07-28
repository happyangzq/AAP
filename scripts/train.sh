#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?Set DATASET_DIR to the dataset root.}"
: "${SAM_CHECKPOINT:?Set SAM_CHECKPOINT to sam_vit_h_4b8939.pth.}"

deepspeed train.py \
  --dataset-dir "${DATASET_DIR}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --version xinlai/LISA-7B-v1 \
  --vision-tower openai/clip-vit-large-patch14 \
  --precision bf16 \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-4 \
  --anchor-loss-weight 3.0 \
  --target-layer 16 \
  --output-dir runs/aap
