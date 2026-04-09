#!/bin/bash

set -euo pipefail

CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/checkpoint.ckpt}
CUDA_DEVICE=${CUDA_DEVICE:-2}
# SCENE_NAME includes: Tablet, Stone, Man, Dog, Sculpture, Durian
SCENE_NAME=${SCENE_NAME:-Tablet}
CONTEXT_VIEWS=${CONTEXT_VIEWS:-0,4}
TARGET_VIEWS=${TARGET_VIEWS:-0,2}

scene_name_handle() {
  case "$1" in
    Tablet|Stone|Man|Dog|Sculpture|Durian)
      printf "%s" "$1"
      ;;
    *)
      echo "Unsupported BlendMVS scene: $1" >&2
      echo "Allowed: Tablet, Stone, Man, Dog, Sculpture, Durian" >&2
      exit 1
      ;;
  esac
}

SCENE_NAME=$(scene_name_handle "$SCENE_NAME")

GENERATE_2D_PLY_OUTPUT_DATASET=data \
GENERATE_2D_PLY_SCENE_NAME="$SCENE_NAME" \
GENERATE_2D_PLY_CONTEXT_VIEWS="$CONTEXT_VIEWS" \
GENERATE_2D_PLY_TARGET_VIEWS="$TARGET_VIEWS" \
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m src.paper.generate_2d_ply \
  +experiment=re10k \
  checkpointing.load="$CHECKPOINT_PATH"
