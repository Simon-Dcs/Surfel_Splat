#!/bin/bash

set -euo pipefail

CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/checkpoint.ckpt}
CUDA_DEVICE=${CUDA_DEVICE:-2}
SCENE_ID=${SCENE_ID:-24}
CONTEXT_VIEWS=${CONTEXT_VIEWS:-0,2,9}
TARGET_VIEWS=${TARGET_VIEWS:-4,2}
SCENE_STAGE=${SCENE_STAGE:-train}

scene_name_handle() {
  printf "scan%s_%s" "$1" "$2"
}

SCENE_NAME=$(scene_name_handle "$SCENE_ID" "$SCENE_STAGE")

GENERATE_2D_PLY_OUTPUT_DATASET=data \
GENERATE_2D_PLY_SCENE_NAME="$SCENE_NAME" \
GENERATE_2D_PLY_CONTEXT_VIEWS="$CONTEXT_VIEWS" \
GENERATE_2D_PLY_TARGET_VIEWS="$TARGET_VIEWS" \
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.paper.generate_2d_ply \
  +experiment=re10k \
  checkpointing.load="$CHECKPOINT_PATH"
