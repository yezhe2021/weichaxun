#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
DATA_ROOT="${DATA_ROOT:-${P2A2_ROOT}/data}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${ROOT}/cache_qwen3_4b_native_kv_pairs}"
AUDIT_OUT="${AUDIT_OUT:-${ROOT}/audit}"
TRAIN_OUT="${TRAIN_OUT:-${ROOT}/train_writer_qwen3_4b}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/eval_writer_qwen3_4b}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

case "${1:-help}" in
  audit)
    require_file "${DATA_ROOT}/test.jsonl"
    require_file "${READER_CHECKPOINT}"
    "${PY}" "${ROOT}/audit_p2b.py" \
      --sender-model "${SENDER_MODEL}" \
      --receiver-model "${RECEIVER_MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --reader-checkpoint "${READER_CHECKPOINT}" \
      --out "${AUDIT_OUT}"
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
  train)
    require_file "${SENDER_CACHE_ROOT}/train/index.json"
    require_file "${TEACHER_CACHE_ROOT}/train/index.json"
    require_file "${READER_CHECKPOINT}"
    "${PY}" "${ROOT}/train_p2b_writer.py" \
      --receiver-model "${RECEIVER_MODEL}" \
      --reader-checkpoint "${READER_CHECKPOINT}" \
      --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
      --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
      --out "${TRAIN_OUT}" \
      --max-pairs "${TRAIN_PAIRS}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  eval)
    require_file "${SENDER_CACHE_ROOT}/test/index.json"
    require_file "${TEACHER_CACHE_ROOT}/test/index.json"
    require_file "${READER_CHECKPOINT}"
    require_file "${TRAIN_OUT}/checkpoint_latest.pt"
    "${PY}" "${ROOT}/eval_p2b_writer.py" \
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
    bash "${ROOT}/run_all.sh" audit
    bash "${ROOT}/run_all.sh" cache-sender-train
    bash "${ROOT}/run_all.sh" cache-sender-test
    bash "${ROOT}/run_all.sh" train
    bash "${ROOT}/run_all.sh" eval
    ;;
  *)
    cat <<'USAGE'
P2-B complete run:
  bash run_all.sh all

Individual stages:
  bash run_all.sh audit
  bash run_all.sh cache-sender-train
  bash run_all.sh cache-sender-test
  bash run_all.sh train
  bash run_all.sh eval
USAGE
    ;;
esac
