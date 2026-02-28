#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m ministral_ft.train \
  --train-file data/examples/train.jsonl \
  --eval-file data/examples/valid.jsonl \
  --output-dir runs/ministral-3b-lora \
  "$@"
