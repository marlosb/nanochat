#!/bin/bash

set -euo pipefail

# Resume pipeline from an interrupted base_train without re-running setup,
# dataset downloads, or tokenizer training.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. This resume script assumes setup already exists."
    exit 1
fi

source .venv/bin/activate

OUTPUT_DIRNAME="d24"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"

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

echo ""
echo "#### Resuming base_train from step $RESUME_STEP_NUM"
echo "#### Checkpoint dir: $CHECKPOINT_DIR"
echo ""
python -m scripts.base_train \
    --depth=24 \
    --total_batch_size=32768 \
    --sample_every=100000 \
    --save_every=100000 \
    --run=march \
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
python -m scripts.base_eval --model_tag="$OUTPUT_DIRNAME"

echo ""
echo "#### base_eval complete"
echo "#### starting mid_train"
echo ""

# NOTE: ensure the same device_batch_size policy you used originally.
python -m scripts.mid_train --device_batch_size=24 --eval_tokens=32768

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
