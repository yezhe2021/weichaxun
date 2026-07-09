#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"

SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
DATA="${DATA:-/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/self_traces/receiver_self_traces_256.jsonl}"
CHECKPOINT="${CHECKPOINT:-/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/train_paper_sweep/256_e1e5/checkpoint_final.pt}"

MAX_SAMPLES="${MAX_SAMPLES:-16}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

"${PY}" "${ROOT}/run_teacher_forced_vs_free.py" \
  --sender-model "${SENDER}" \
  --receiver-model "${RECEIVER}" \
  --data "${DATA}" \
  --adapter-checkpoint "${CHECKPOINT}" \
  --out "${ROOT}/results" \
  --max-samples "${MAX_SAMPLES}" \
  --max-source-tokens "${MAX_SOURCE_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"
