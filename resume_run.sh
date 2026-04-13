#!/bin/bash

set -euo pipefail

# Resume pipeline from an interrupted base_train.
# This script bootstraps env, downloads required data files, then resumes training.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo ".venv not found. Creating virtual environment..."
    uv venv
fi

echo "Syncing dependencies into .venv..."
uv sync --extra gpu

source .venv/bin/activate

echo "Downloading identity_conversations.jsonl..."
curl -fL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    "https://huggingface.co/datasets/marlosb/auxiliary_data/resolve/main/identity_conversations.jsonl"

OUTPUT_DIRNAME="d24"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"
TARGET_NUM_SHARDS="${TARGET_NUM_SHARDS:-1400}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
DOWNLOAD_START_IDX="${DOWNLOAD_START_IDX:-640}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Set NANOCHAT_BASE_DIR/MODEL_TAG/DEPTH to match your previous run."
    exit 1
fi

LAST_MODEL_FILE="$(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort | tail -n 1 || true)"
if [ -z "$LAST_MODEL_FILE" ]; then
    echo "ERROR: no model checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

RESUME_STEP="$(basename "$LAST_MODEL_FILE" | sed -E 's/^model_([0-9]{6})\.pt$/\1/')"
if ! [[ "$RESUME_STEP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse resume step from $LAST_MODEL_FILE"
    exit 1
fi

RESUME_STEP_NUM=$((10#$RESUME_STEP))

if ! [[ "$DOWNLOAD_START_IDX" =~ ^[0-9]+$ ]]; then
    DOWNLOAD_START_IDX=640
fi

echo ""
echo "#### Downloading shards in range: $DOWNLOAD_START_IDX..$((TARGET_NUM_SHARDS - 1))"
echo ""
python -m nanochat.dataset -n "$TARGET_NUM_SHARDS" -s "$DOWNLOAD_START_IDX" -w "$DOWNLOAD_WORKERS"

echo ""
echo "#### Resuming base_train from step $RESUME_STEP_NUM"
echo "#### Checkpoint dir: $CHECKPOINT_DIR"
echo ""
python -m scripts.base_train \
    --depth=24 \
    --device_batch_size=24 \
    --total_batch_size=49152 \
    --target_param_data_ratio=40 \
    --sample_every=50000 \
    --save_every=50000 \
    --run=march \
    --eval_every=10000 \
    --core_metric_every=10000 \
    --resume_from_step="$RESUME_STEP_NUM"

echo ""
echo "#### base_train complete"
echo "#### starting base_loss"
echo ""
python -m scripts.base_loss --split_tokens=16384 --model_tag="$OUTPUT_DIRNAME"

echo ""
echo "#### base_loss complete"
echo "#### starting base_eval"
echo ""
python -m scripts.base_eval

echo ""
echo "#### base_eval complete"
echo "#### starting mid_train"
echo ""

# NOTE: ensure the same device_batch_size policy you used originally.
python -m scripts.mid_train --device_batch_size=8 --eval_tokens=32768

echo ""
echo "#### mid_train complete"
echo "#### starting mid_eval"
echo ""
python -m scripts.chat_eval -i mid

echo ""
echo "#### mid_eval complete"
echo "#### starting sft"
echo ""
python -m scripts.chat_sft

echo ""
echo "#### sft complete"
echo "#### starting sft_eval"
echo ""
python -m scripts.chat_eval -i sft

echo ""
echo "#### sft_eval complete"
echo "#### starting report generation"
echo ""
python -m nanochat.report generate

echo ""
echo "#### Resume pipeline complete"
