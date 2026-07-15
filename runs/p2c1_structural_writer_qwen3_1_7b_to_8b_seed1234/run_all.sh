#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
DATA_ROOT="${DATA_ROOT:-${P2A2_ROOT}/data}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${ROOT}/cache_qwen3_1_7b_native_kv_pairs}"
AUDIT_ROOT="${AUDIT_ROOT:-${ROOT}/audit}"
TRAIN_OUT="${TRAIN_OUT:-${ROOT}/train_writer_qwen3_1_7b}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/eval_writer_qwen3_1_7b}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

audit_split() {
  local split="$1"
  require_file "${DATA_ROOT}/${split}.jsonl"
  require_file "${READER_CHECKPOINT}"
  "${PY}" "${ROOT}/audit_p2c1.py" \
    --sender-model "${SENDER_MODEL}" \
    --receiver-model "${RECEIVER_MODEL}" \
    --data "${DATA_ROOT}/${split}.jsonl" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --out "${AUDIT_ROOT}/${split}"
}

case "${1:-help}" in
  audit-train)
    audit_split train
    ;;
  audit-test)
    audit_split test
    ;;
  cache-sender-train)
    require_file "${DATA_ROOT}/train.jsonl"
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${SENDER_MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --out "${SENDER_CACHE_ROOT}/train" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-sender-test)
    require_file "${DATA_ROOT}/test.jsonl"
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${SENDER_MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${SENDER_CACHE_ROOT}/test" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  audit-cache)
    require_file "${SENDER_CACHE_ROOT}/train/index.json"
    require_file "${TEACHER_CACHE_ROOT}/train/index.json"
    "${PY}" "${ROOT}/audit_cache_geometry.py" \
      --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
      --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
      --out "${AUDIT_ROOT}/cache_geometry"
    ;;
  train)
    require_file "${SENDER_CACHE_ROOT}/train/index.json"
    require_file "${TEACHER_CACHE_ROOT}/train/index.json"
    require_file "${READER_CHECKPOINT}"
    "${PY}" "${ROOT}/train_p2c1_writer.py" \
      --receiver-model "${RECEIVER_MODEL}" \
      --reader-checkpoint "${READER_CHECKPOINT}" \
      --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
      --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
      --out "${TRAIN_OUT}" \
      --max-pairs "${TRAIN_PAIRS}" \
      --epochs "${EPOCHS}" \
      --layer-width 5 \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  eval)
    require_file "${SENDER_CACHE_ROOT}/test/index.json"
    require_file "${TEACHER_CACHE_ROOT}/test/index.json"
    require_file "${READER_CHECKPOINT}"
    require_file "${TRAIN_OUT}/checkpoint_latest.pt"
    "${PY}" "${ROOT}/eval_p2c1_writer.py" \
      --receiver-model "${RECEIVER_MODEL}" \
      --reader-checkpoint "${READER_CHECKPOINT}" \
      --writer-checkpoint "${TRAIN_OUT}/checkpoint_latest.pt" \
      --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
      --teacher-index "${TEACHER_CACHE_ROOT}/test/index.json" \
      --out "${EVAL_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  all)
    bash "${ROOT}/run_all.sh" audit-train
    bash "${ROOT}/run_all.sh" audit-test
    bash "${ROOT}/run_all.sh" cache-sender-train
    bash "${ROOT}/run_all.sh" cache-sender-test
    bash "${ROOT}/run_all.sh" audit-cache
    bash "${ROOT}/run_all.sh" train
    bash "${ROOT}/run_all.sh" eval
    ;;
  *)
    cat <<'USAGE'
P2-C1 complete run:
  bash run_all.sh all

Individual stages:
  bash run_all.sh audit-train
  bash run_all.sh audit-test
  bash run_all.sh cache-sender-train
  bash run_all.sh cache-sender-test
  bash run_all.sh audit-cache
  bash run_all.sh train
  bash run_all.sh eval
USAGE
    ;;
esac
