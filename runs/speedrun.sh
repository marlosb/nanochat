#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank single H100 GPU node.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

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
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
run_cmd python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Pretraining datasets

# Download all shards for both datasets before training.
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2
run_cmd python -m nanochat.dataset --dataset gigaverbo-v2-synth

# -----------------------------------------------------------------------------
# Base model (pretraining)

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run="$WANDB_RUN"
# evaluate the model: CORE metric, BPB on train/val, and draw samples
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
run_cmd curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://huggingface.co/datasets/marlosb/auxiliary_data/resolve/main/identity_conversations.jsonl

# run SFT and eval the model
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- --device-batch-size=16 --run="$WANDB_RUN"
run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
run_cmd python -m nanochat.report generate
