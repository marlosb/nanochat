#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank single H100 GPU node and takes longer than speedrun.sh
# because it runs an additional base pretraining stage on gigaverbo-v2-synth.

# 1) Example launch (simplest):
# bash runs/speedrun_v2_then_synth.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/speedrun_v2_then_synth.log -S speedrun_v2_then_synth bash runs/speedrun_v2_then_synth.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun_v2_then_synth screen -L -Logfile runs/speedrun_v2_then_synth.log -S speedrun_v2_then_synth bash runs/speedrun_v2_then_synth.sh

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

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-19}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-38912}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-100}"
SYNTH_TARGET_PARAM_DATA_RATIO="${SYNTH_TARGET_PARAM_DATA_RATIO:-2.35}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
run_cmd mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    run_cmd sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || run_cmd uv venv
# install the repo dependencies
run_cmd uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
print_divider
echo "[RUN] source .venv/bin/activate"
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=june
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
run_cmd python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Pretraining datasets

# Download 57 shards for gigaverbo-v2 and 10 shards for gigaverbo-v2-synth before training.
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2 -n 57
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2-synth -n 10

# -----------------------------------------------------------------------------
# Base model (pretraining) - stage 1 on gigaverbo-v2

# d24 model tuned for single H100 runs.
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- --depth=24 --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" --fp8 --dataset gigaverbo-v2 --run="$WANDB_RUN"
# evaluate the model: CORE metric, BPB on train/val, and draw samples
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- --device-batch-size="$DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# Base model (pretraining) - stage 2 on gigaverbo-v2-synth

# Resume from the latest d24 checkpoint and continue for synth-token-aligned steps.
LAST_BASE_STEP=$(python - <<'PY'
import os
from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir
checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", "d24")
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
print_divider
echo "[INFO] Resuming d24 from step ${LAST_BASE_STEP} to ${SYNTH_END_STEP} on gigaverbo-v2-synth (stage-2 ratio: ${SYNTH_TARGET_PARAM_DATA_RATIO})"
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- --depth=24 --target-param-data-ratio="$SYNTH_TARGET_PARAM_DATA_RATIO" --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" --fp8 --dataset gigaverbo-v2-synth --model-tag d24 --resume-model-tag d24 --resume-from-step "${LAST_BASE_STEP}" --num-iterations "${SYNTH_END_STEP}" --run="$WANDB_RUN"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
run_cmd curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://huggingface.co/datasets/marlosb/auxiliary_data/resolve/main/identity_conversations.jsonl

# run SFT and eval the model
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- --device-batch-size="$DEVICE_BATCH_SIZE" --run="$WANDB_RUN"
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
run_cmd python -m nanochat.report generate
