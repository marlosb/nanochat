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
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://huggingface.co/datasets/marlosb/auxiliary_data/resolve/main/identity_conversations.jsonl

# train tokenizer on ~4B characters and kick off download of the rest for pretraining
python -m nanochat.dataset -n 16
# start downloading the rest of the shards for a total of 800 (see below why 800)
python -m nanochat.dataset -n 20 &
# todo: download the rest of it
python -m scripts.tok_train 
python -m scripts.tok_eval

# This is a quickly test run, only to make sure all scrips will run fine after
# modifing datasets used. The intetion is to run quickly in a single GPU server.
# Testing parameters:
# Vocab size: 65,536
# num_layers: 8
# model_dim: 512
# num_heads: 4
# num_kv_heads: 4
# Tokens / micro-batch / rank: 8 x 512 = 4,096
# Tokens / micro-batch: 4,096
# Total batch size 16,384 => gradient accumulation steps: 4
# Number of parameters:  92,274,688
# Estimated FLOPs per token: ~ 380,000,000
# Calculated number of iterations from target data:param ratio: 
# Total number of training tokens: 2,768,240,640
# Tokens : Params ratio: 30.00
# Total training FLOPs estimate: 1.044976e+18
# ...

echo ""
echo "#### Starting base_train"
echo ""
python -m scripts.base_train --depth=8 --total_batch_size=16384 --max_seq_len=512 --sample_every=100000 --save_every=100000
echo ""
echo "#### base_train complete"
echo "#### starting base_loss"
echo ""
python -m scripts.base_loss --split_tokens=16384
echo ""
echo "#### base_loss complete "
echo "#### starting base_eval"
echo ""
python -m scripts.base_eval
echo ""
echo "#### base_eval complete"
echo "#### starting mid_train"
echo ""

# midtrain
# NOTE: ensure that we use the same device_batch_size here as the base training script.
python -m scripts.mid_train --device_batch_size=8 --max_seq_len=512 --eval_tokens=16384
echo ""
echo "#### mid_train complete"
echo "#### starting mid_eval"
echo ""
python -m scripts.chat_eval -i mid
echo "#### mid_eval complete"
echo "#### starting sft"
echo ""

# sft
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

# generate final report
python -m nanochat.report generate