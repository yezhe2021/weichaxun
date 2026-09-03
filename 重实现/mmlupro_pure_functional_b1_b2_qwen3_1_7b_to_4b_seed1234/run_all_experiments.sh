#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/重实现/mmlupro_pure_functional_b1_b2_qwen3_1_7b_to_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"

cd "$ROOT"

run_step() {
  printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
  "$@"
  printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step "$PYTHON" -u run_unit_tests.py
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" prepare
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" audit
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" calibrate
run_step "$PYTHON" -u run_gpu_tests.py

# Capacity/code validation on the same 16 samples. B1/B2 independently reload overfit Stage-A best.
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" overfit --writer d2 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer d2 --functional-mode both --checkpoint checkpoints/overfit/d2/stage_a/best.pt
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" overfit --writer d2 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" overfit --writer d2 --stage b --functional-mode all
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope overfit

# Formal 128/32/128 experiment. B1/B2 independently reload formal Stage-A best.
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer d2 --functional-mode both --checkpoint checkpoints/quick/d2/stage_a/best.pt
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage b --functional-mode all
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope formal

printf '[%s] ALL PURE-FUNCTIONAL EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
