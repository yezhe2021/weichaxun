#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_stageb_checkpoint_path_qwen3_4b_to_8b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"
cd "$ROOT"
run_step() { printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; "$@"; printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
run_step "$PYTHON" -u run_unit_tests.py
run_step "$PYTHON" -u audit_cache_reuse.py --config "$CONFIG"
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full36_head8 --stage b --functional-mode final
run_step "$PYTHON" -u evaluate_stageb_checkpoints.py --config "$CONFIG"
printf '[%s] ALL STAGE-B CHECKPOINT-PATH EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
