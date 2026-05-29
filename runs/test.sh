#!/bin/bash

# Test run for a single GPU using the same core training values as speedrun_v2.sh.

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

run_with_timeout() {
    local minutes="$1"
    shift
    if command -v timeout &> /dev/null; then
        print_divider
        echo "[RUN] timeout --signal=INT ${minutes}m $*"
        set +e
        timeout --signal=INT "${minutes}m" "$@"
        status=$?
        set -e
        if [ "$status" -eq 0 ] || [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
            [ "$status" -eq 0 ] || echo "[INFO] Command stopped after timeout (${minutes}m)."
            return 0
        fi
        return "$status"
    fi
    echo "[WARN] 'timeout' not found; running without time cap."
    run_cmd "$@"
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
RUN_MINUTES="${RUN_MINUTES:-120}"
DEPTH="${DEPTH:-24}"
MODEL_TAG="${MODEL_TAG:-test-d24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-19}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-38912}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-100}"
SYNTH_TARGET_PARAM_DATA_RATIO="${SYNTH_TARGET_PARAM_DATA_RATIO:-2.35}"

# Tiny data download (small sample) for both pretraining datasets.
# Validation shard is downloaded automatically by nanochat.dataset.
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2 -n 57
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2-synth -n 10

# Base pretraining stage 1 (same values as speedrun_v2 defaults), capped to 2 hours.
run_with_timeout "$RUN_MINUTES" torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --dataset gigaverbo-v2 \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run="$WANDB_RUN"

# Base pretraining stage 2 on synth, resuming from stage 1, capped to 2 hours.
LAST_BASE_STEP=$(python - <<'PY'
import os
from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir
checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", "test-d24")
print(find_last_step(checkpoint_dir))
PY
)
SYNTH_EXTRA_STEPS=$(python - <<PY
import math
target_ratio = float("${SYNTH_TARGET_PARAM_DATA_RATIO}")
total_batch_size = int("${TOTAL_BATCH_SIZE}")
# d24 scaling params used by base_train logs:
# transformer_matrices (679,478,976) + lm_head (100,663,296)
scaling_params = 780_142_272
target_tokens = target_ratio * scaling_params
print(math.ceil(target_tokens / total_batch_size))
PY
)
SYNTH_END_STEP=$((LAST_BASE_STEP + SYNTH_EXTRA_STEPS))
run_with_timeout "$RUN_MINUTES" torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --dataset gigaverbo-v2-synth \
    --target-param-data-ratio="$SYNTH_TARGET_PARAM_DATA_RATIO" \
    --model-tag="$MODEL_TAG" \
    --resume-model-tag="$MODEL_TAG" \
    --resume-from-step="$LAST_BASE_STEP" \
    --num-iterations="$SYNTH_END_STEP" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run="$WANDB_RUN"

# SFT stage on top of the base checkpoint, capped to 2 hours.
run_with_timeout "$RUN_MINUTES" torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=-1 \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

echo "Test run completed."
