#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m ministral_ft.build_case_corpus \
  --w5-root ../w5 \
  --output-dir data/succession_e2e \
  "$@"
