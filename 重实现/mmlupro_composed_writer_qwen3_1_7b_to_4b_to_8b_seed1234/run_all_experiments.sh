#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_composed_writer_qwen3_1_7b_to_4b_to_8b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"
cd "$ROOT"
run_step() { printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; "$@"; printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
run_step "$PYTHON" -u evaluate_composition.py --config "$CONFIG" --scope smoke --limit 2
run_step "$PYTHON" -u evaluate_composition.py --config "$CONFIG" --scope formal
printf '[%s] ALL COMPOSED-WRITER EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
