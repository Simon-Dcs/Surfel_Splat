#!/bin/bash

set -euo pipefail

CUDA_DEVICES=${CUDA_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-1}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m src.main \
  +experiment=re10k \
  data_loader.train.batch_size="$BATCH_SIZE"
