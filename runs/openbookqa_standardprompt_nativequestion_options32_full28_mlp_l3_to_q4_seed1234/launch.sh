#!/usr/bin/env bash
set -u
ROOT="/home/yezhe/异构模型/mmlupro_standardprompt_nativequestion_optionskv_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODE="${1:-study}"
cd "$ROOT" || exit 1
mkdir -p logs
"$PYTHON" -u tests.py
"$PYTHON" -u run_pipeline.py --mode "$MODE" all
