#!/bin/bash

# The $1000 tier of nanochat
# Designed to run end-to-end for $1000/24 ~= 41.6 hours on an 8XH100 node
# A bit sparser on comments, see speedrun.sh for more detail

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# train tokenizer on ~4B characters and kick off download of the rest for pretraining
python -m nanochat.dataset -n 16
# start downloading the rest of the shards for a total of 800 (see below why 800)
#python -m nanochat.dataset -n 800 &
# todo: download the rest of it
python -m scripts.tok_train --vocab_size=30000
python -m scripts.tok_eval