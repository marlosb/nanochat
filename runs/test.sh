#!/bin/bash

# Quick smoke test for a single GPU.
# Runs a tiny base pretraining + tiny SFT flow to validate the pipeline executes.

set -euo pipefail

print_divider() {
    echo ""
    echo "------------------------------------------------------------"
}

run_cmd() {
    print_divider
    echo "[RUN] $*"
    "$@"
}

export OMP_NUM_THREADS=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
run_cmd mkdir -p "$NANOCHAT_BASE_DIR"

# Optional setup (skip with SKIP_SETUP=1 if .venv is already ready)
if [ -z "${SKIP_SETUP:-}" ]; then
    if ! command -v uv &> /dev/null; then
        run_cmd sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
    [ -d ".venv" ] || run_cmd uv venv
    run_cmd uv sync --extra gpu
fi
print_divider
echo "[RUN] source .venv/bin/activate"
source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-dummy}"

# Tiny data download (small sample) for both pretraining datasets.
# Validation shard is downloaded automatically by nanochat.dataset.
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2 -n 2
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2-synth -n 2

# Tiny base pretraining run (depth 8, very short horizon).
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=8 \
    --dataset gigaverbo-v2 \
    --model-tag=test-d8 \
    --num-iterations=20 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=2048 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run="$WANDB_RUN"

# Tiny base pretraining run on synth, resuming from previous tiny checkpoint
# to follow the same two-stage flow as speedrun_v2.sh.
LAST_BASE_STEP=$(python - <<'PY'
import os
from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir
checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", "test-d8")
print(find_last_step(checkpoint_dir))
PY
)
SYNTH_END_STEP=$((LAST_BASE_STEP * 2))
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=8 \
    --dataset gigaverbo-v2-synth \
    --model-tag=test-d8 \
    --resume-model-tag=test-d8 \
    --resume-from-step="$LAST_BASE_STEP" \
    --num-iterations="$SYNTH_END_STEP" \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=2048 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run="$WANDB_RUN"

# Tiny SFT run on top of the tiny base checkpoint.
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag=test-d8 \
    --num-iterations=20 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=2048 \
    --eval-every=-1 \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

echo "Smoke test completed."
