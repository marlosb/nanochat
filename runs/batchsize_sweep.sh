#!/bin/bash

# Sweep depth=24 base pretraining configs to compare tok/sec and MFU on a single GPU.
# Each config runs for ~2 hours (via timeout when available) and writes a separate log.

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

run_cmd_no_fail() {
    print_divider
    echo "[RUN] $*"
    set +e
    "$@"
    status=$?
    set -e
    return $status
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
DATASET="${DATASET:-gigaverbo-v2}"
RUN_MINUTES="${RUN_MINUTES:-120}"
DEPTH="${DEPTH:-24}"
MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-sweep-d24}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000000}" # high cap; timeout is expected to stop first

LOG_DIR="$NANOCHAT_BASE_DIR/batchsize_sweep_logs_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$LOG_DIR/summary.tsv"
run_cmd mkdir -p "$LOG_DIR"

print_divider
echo "[INFO] Logs directory: $LOG_DIR"
echo -e "device_batch_size\ttotal_batch_size\tstatus\tlast_tok_per_sec\tlast_mfu\tlog_file" > "$SUMMARY_FILE"

CONFIGS=(
    "19 32768"
    "19 38912"
    "19 49152"
    "19 65536"
)

for cfg in "${CONFIGS[@]}"; do
    read -r DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE <<< "$cfg"
    MODEL_TAG="${MODEL_TAG_PREFIX}-dbs${DEVICE_BATCH_SIZE}-tbs${TOTAL_BATCH_SIZE}"
    LOG_FILE="$LOG_DIR/${MODEL_TAG}.log"

    print_divider
    echo "[INFO] Starting config: --device-batch-size=${DEVICE_BATCH_SIZE} --total-batch-size=${TOTAL_BATCH_SIZE}"
    echo "[INFO] Logging to: $LOG_FILE"

    CMD=(
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train --
        --depth="$DEPTH"
        --dataset "$DATASET"
        --model-tag="$MODEL_TAG"
        --run="$WANDB_RUN"
        --num-iterations="$NUM_ITERATIONS"
        --device-batch-size="$DEVICE_BATCH_SIZE"
        --total-batch-size="$TOTAL_BATCH_SIZE"
        --eval-every=-1
        --core-metric-every=-1
        --sample-every=-1
        --save-every=-1
    )

    status_text="ok"
    if command -v timeout &> /dev/null; then
        if run_cmd_no_fail timeout --signal=INT "${RUN_MINUTES}m" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
            status_text="completed"
        else
            exit_code=${PIPESTATUS[0]}
            if [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 130 ]; then
                status_text="timed_out"
            else
                status_text="failed_exit_${exit_code}"
            fi
        fi
    else
        echo "[WARN] 'timeout' not found; running fixed iterations without time cap."
        if run_cmd_no_fail "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
            status_text="completed"
        else
            exit_code=${PIPESTATUS[0]}
            status_text="failed_exit_${exit_code}"
        fi
    fi

    if grep -qiE "out of memory|cuda error: out of memory|cublas.*alloc|cuda out of memory" "$LOG_FILE"; then
        status_text="oom"
    fi

    last_tok_per_sec=$(grep -Eo "tok/sec: [0-9,]+" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    last_mfu=$(grep -Eo "(bf16_)?mfu: [0-9]+(\.[0-9]+)?" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    last_tok_per_sec="${last_tok_per_sec:-na}"
    last_mfu="${last_mfu:-na}"

    echo -e "${DEVICE_BATCH_SIZE}\t${TOTAL_BATCH_SIZE}\t${status_text}\t${last_tok_per_sec}\t${last_mfu}\t${LOG_FILE}" >> "$SUMMARY_FILE"
    echo "[RESULT] ${MODEL_TAG}: status=${status_text} tok/sec=${last_tok_per_sec} mfu=${last_mfu}"
done

print_divider
echo "[DONE] Sweep finished."
echo "[DONE] Summary: $SUMMARY_FILE"
