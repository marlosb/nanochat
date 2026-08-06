#!/bin/bash

# Continue a completed runs/base_train.sh d24 checkpoint on all available
# gigaverbo-v2-synth training shards.
#
# Example:
# WANDB_RUN=extended screen -L -Logfile runs/extended_base_train.log \
#   -S extended_base_train bash runs/extended_base_train.sh

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
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-19}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-77824}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-60000}"
EVAL_EVERY="${EVAL_EVERY:-20000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10000}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-10000}"
# 2.35 covers the original 10-shard subset; 8.70 scales it to all 37
# training shards. The 38th and final shard is reserved for validation.
SYNTH_TARGET_PARAM_DATA_RATIO="${SYNTH_TARGET_PARAM_DATA_RATIO:-8.70}"
# Decay throughout the short continuation instead of adding another plateau.
SYNTH_WARMDOWN_RATIO="${SYNTH_WARMDOWN_RATIO:-1.0}"
WANDB_RUN="${WANDB_RUN:-extended}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

if [ ! -d ".venv" ]; then
    echo "Missing .venv. Run runs/base_train.sh before this continuation script." >&2
    exit 1
fi

print_divider
echo "[RUN] source .venv/bin/activate"
source .venv/bin/activate

# Download and validate the complete dataset before constructing the training
# loader, so its first epoch includes every training shard.
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2-synth -n -1
run_cmd python -c "import os; import pyarrow.parquet as pq; from nanochat.dataset import DATASET_SPECS, index_to_filename, list_parquet_files; tag='gigaverbo-v2-synth'; count=DATASET_SPECS[tag]['shard_count']; paths=list_parquet_files(dataset_tag=tag); expected=[index_to_filename(i, dataset_tag=tag) for i in range(count)]; actual=[os.path.basename(path) for path in paths]; assert actual == expected, f'Expected all {count} ordered shards, found {len(actual)}: {actual}'; [pq.ParquetFile(path).metadata for path in paths]; print(f'Validated {count - 1} training shards and 1 validation shard')"

LAST_BASE_STEP=$(python -c "import os; from nanochat.checkpoint_manager import find_last_step; from nanochat.common import get_base_dir; print(find_last_step(os.path.join(get_base_dir(), 'base_checkpoints', 'd24')))")
SYNTH_EXTRA_STEPS=$(python -c "import math; print(math.ceil(float('${SYNTH_TARGET_PARAM_DATA_RATIO}') * 780142272 / int('${TOTAL_BATCH_SIZE}')))")
SYNTH_END_STEP=$((LAST_BASE_STEP + SYNTH_EXTRA_STEPS))

print_divider
echo "[INFO] Resuming d24 from step ${LAST_BASE_STEP} to ${SYNTH_END_STEP} on all gigaverbo-v2-synth training data"
echo "[INFO] Synthetic ratio: ${SYNTH_TARGET_PARAM_DATA_RATIO}; warmdown ratio: ${SYNTH_WARMDOWN_RATIO}"

run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=24 \
    --num-iterations="$SYNTH_END_STEP" \
    --target-param-data-ratio="$SYNTH_TARGET_PARAM_DATA_RATIO" \
    --warmdown-ratio="$SYNTH_WARMDOWN_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --save-every="$CHECKPOINT_EVERY" \
    --fp8 \
    --dataset gigaverbo-v2-synth \
    --model-tag=d24 \
    --resume-model-tag=d24 \
    --resume-from-step="$LAST_BASE_STEP" \
    --run="$WANDB_RUN"

run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE"

run_cmd python -m nanochat.report generate
