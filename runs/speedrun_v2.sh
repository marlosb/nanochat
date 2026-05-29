#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes longer than speedrun.sh
# because it runs an additional base pretraining stage on gigaverbo-v2-synth.

# 1) Example launch (simplest):
# bash runs/speedrun_v2_then_synth.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/speedrun_v2_then_synth.log -S speedrun_v2_then_synth bash runs/speedrun_v2_then_synth.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun_v2_then_synth screen -L -Logfile runs/speedrun_v2_then_synth.log -S speedrun_v2_then_synth bash runs/speedrun_v2_then_synth.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first few shards of the Gigaverbo-v2 pretraining dataset
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) - stage 1 on gigaverbo-v2
echo "Waiting for gigaverbo-v2 dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --dataset gigaverbo-v2 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# Base model (pretraining) - stage 2 on gigaverbo-v2-synth

# Download synth shards before the second base pretraining stage
python -m nanochat.dataset --dataset gigaverbo-v2-synth -n 170

# Resume from the latest d24 checkpoint and continue for an additional matching number of steps
LAST_BASE_STEP=$(python - <<'PY'
import os
from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir
checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", "d24")
print(find_last_step(checkpoint_dir))
PY
)
SYNTH_END_STEP=$((LAST_BASE_STEP * 2))
echo "Resuming d24 from step ${LAST_BASE_STEP} and continuing to step ${SYNTH_END_STEP} on gigaverbo-v2-synth..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --dataset gigaverbo-v2-synth --model-tag d24 --resume-model-tag d24 --resume-from-step ${LAST_BASE_STEP} --num-iterations ${SYNTH_END_STEP} --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
