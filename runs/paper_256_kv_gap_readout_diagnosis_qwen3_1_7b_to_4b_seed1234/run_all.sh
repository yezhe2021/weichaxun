#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"

SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
DATA="${DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
CHECKPOINT="${CHECKPOINT:-/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/train_paper_sweep/256_e1e5/checkpoint_final.pt}"

MAX_SAMPLES="${MAX_SAMPLES:-32}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-128}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

case "${1:-run}" in
  run)
    "${PY}" "${ROOT}/diagnose_paper_kv_gap.py" \
      --sender-model "${SENDER}" \
      --receiver-model "${RECEIVER}" \
      --data "${DATA}" \
      --adapter-checkpoint "${CHECKPOINT}" \
      --method-label paper_256_e1e5 \
      --out "${ROOT}/results" \
      --max-samples "${MAX_SAMPLES}" \
      --max-source-tokens "${MAX_SOURCE_TOKENS}" \
      --attention-topk "${ATTENTION_TOPK}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  *)
    echo "Usage: bash $0 run"
    ;;
esac
