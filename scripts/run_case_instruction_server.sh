#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m ministral_ft.case_instruction_server \
  --state-dir data/case_instruction_server \
  --corpus-file data/succession_e2e/e2e_cases.jsonl \
  "$@"
