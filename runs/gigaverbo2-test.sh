#!/bin/bash

set -euo pipefail

# Smoke-test script for Gigaverbo-v2 migration.
# Intended for running on a fresh machine.
#
# Usage:
#   bash runs/gigaverbo2-test.sh

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Install uv and project deps (follow existing run scripts).
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
if command -v nvidia-smi &> /dev/null; then
  uv sync --extra gpu
else
  uv sync --extra cpu
fi
source .venv/bin/activate

# Ensure local parquet data exists for smoke tests.
if [ ! -d "$NANOCHAT_BASE_DIR/base_data_gigaverbo_v2" ] || [ "$(find "$NANOCHAT_BASE_DIR/base_data_gigaverbo_v2" -maxdepth 1 -name '*.parquet' | wc -l)" -eq 0 ]; then
  python -m nanochat.dataset -n 1
fi

# Smoke test #1: tokenizer path on a tiny text budget.
python -m scripts.tok_train --max-chars=50000 --vocab-size=512 --doc-cap=1000

# Smoke test #2: base-train path (dataloader/model/step) with minimal compute.
python -m scripts.base_train \
  --device-type=cpu \
  --depth=2 \
  --max-seq-len=64 \
  --device-batch-size=1 \
  --total-batch-size=64 \
  --num-iterations=1 \
  --target-param-data-ratio=-1 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --run=dummy
