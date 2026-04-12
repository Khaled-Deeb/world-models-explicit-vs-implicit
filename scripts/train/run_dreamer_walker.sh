#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DREAMER_ROOT="${DREAMER_ROOT:-$PROJECT_ROOT/third_party/dreamerv3}"
DREAMER_VENV="${DREAMER_VENV:-$PROJECT_ROOT/.venv_dreamer_gpu}"

SEED="${1:-1}"
STEPS="${2:-100000}"
RUN_NAME="${3:-dreamer_walker_seed${SEED}_${STEPS}}"

cd "$DREAMER_ROOT"
source "$DREAMER_VENV/bin/activate"

mkdir -p "$PROJECT_ROOT/results/runs"

python dreamerv3/main.py \
  --logdir "$PROJECT_ROOT/results/runs/$RUN_NAME" \
  --configs dmc_proprio \
  --task dmc_walker_walk \
  --seed "$SEED" \
  --run.steps "$STEPS" \
  --run.envs 4 \
  --run.train_ratio 8 \
  --run.report_every 10 \
  --run.log_every 5 \
  --run.save_every 15 \
  --run.debug True \
  --jax.debug True \
  --jax.prealloc False
