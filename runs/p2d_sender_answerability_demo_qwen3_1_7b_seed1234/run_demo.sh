#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
DATA="${DATA:-/home/yezhe/伪查询/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/data/test.jsonl}"
OUT="${OUT:-${ROOT}/demo_16_pairs}"
MAX_PAIRS="${MAX_PAIRS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

export TOKENIZERS_PARALLELISM=false

"${PY}" "${ROOT}/sender_answerability_demo.py" \
  --model "${MODEL}" \
  --data "${DATA}" \
  --out "${OUT}" \
  --max-pairs "${MAX_PAIRS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"
