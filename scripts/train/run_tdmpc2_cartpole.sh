#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
TD_ROOT="${TD_ROOT:-$PROJECT_ROOT/third_party/tdmpc2}"
TD_IMAGE="${TD_IMAGE:-tdmpc2:1.0.1}"

SEED="${1:-1}"
STEPS="${2:-200000}"
MODEL_SIZE="${3:-5}"
EXP_NAME="${4:-cartpole_${STEPS}_seed${SEED}}"

docker run --rm -it --gpus all \
  -v "$TD_ROOT:/tdmpc2" \
  "$TD_IMAGE" /bin/bash -lc "
cd /tdmpc2/tdmpc2 && \
mkdir -p /tdmpc2/tdmpc2/logs/cartpole-swingup/${SEED}/${EXP_NAME} && \
python train.py \
  task=cartpole-swingup \
  model_size=${MODEL_SIZE} \
  steps=${STEPS} \
  eval_freq=10000 \
  enable_wandb=false \
  save_video=false \
  exp_name=${EXP_NAME} \
  seed=${SEED} 2>&1 | tee /tdmpc2/tdmpc2/logs/cartpole-swingup/${SEED}/${EXP_NAME}/console.log
"
