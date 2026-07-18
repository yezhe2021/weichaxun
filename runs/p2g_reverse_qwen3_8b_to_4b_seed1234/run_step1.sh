#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
SOURCE_DATA_ROOT="${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/data"
TRAIN_DATA="${SOURCE_DATA_ROOT}/train.jsonl"
TEST_DATA="${SOURCE_DATA_ROOT}/test.jsonl"
TRAIN_CACHE="${ROOT}/step1_native_reader/cache/train"
TEST_CACHE="${ROOT}/step1_native_reader/cache/test"
TRAIN_OUT="${ROOT}/step1_native_reader/train"
EVAL_OUT="${ROOT}/step1_native_reader/eval"
LOG_DIR="${ROOT}/logs"
TRAIN_PAIRS=512
TEST_PAIRS=64
EPOCHS=2

mkdir -p "${LOG_DIR}"

audit() {
  "${PY}" "${SCRIPT_DIR}/audit_p2g1.py" \
    --model "${MODEL}" \
    --train-data "${TRAIN_DATA}" \
    --test-data "${TEST_DATA}" \
    --out "${ROOT}/step1_native_reader/AUDIT.json"
}

cache_train() {
  if [[ ! -f "${TRAIN_CACHE}/CACHE_SUCCESS.json" ]]; then
    "${PY}" "${SCRIPT_DIR}/cache_qwen3_4b_native_kv.py" \
      --model "${MODEL}" --data "${TRAIN_DATA}" --out "${TRAIN_CACHE}" \
      --max-samples "$((TRAIN_PAIRS * 2))" --device cuda --dtype float16
  fi
}

cache_test() {
  if [[ ! -f "${TEST_CACHE}/CACHE_SUCCESS.json" ]]; then
    "${PY}" "${SCRIPT_DIR}/cache_qwen3_4b_native_kv.py" \
      --model "${MODEL}" --data "${TEST_DATA}" --out "${TEST_CACHE}" \
      --max-samples "$((TEST_PAIRS * 2))" --device cuda --dtype float16
  fi
}

train() {
  if [[ ! -f "${TRAIN_OUT}/TRAIN_SUCCESS.json" ]]; then
    "${PY}" "${SCRIPT_DIR}/train_qwen3_4b_native_reader.py" \
      --model "${MODEL}" --train-index "${TRAIN_CACHE}/index.json" --out "${TRAIN_OUT}" \
      --config-name query_only --query-rank 32 --output-rank 0 \
      --max-pairs "${TRAIN_PAIRS}" --epochs "${EPOCHS}" --device cuda --dtype float16
  fi
}

evaluate() {
  if [[ ! -f "${EVAL_OUT}/SUCCESS.json" ]]; then
    "${PY}" "${SCRIPT_DIR}/eval_qwen3_4b_native_reader.py" \
      --model "${MODEL}" --test-index "${TEST_CACHE}/index.json" \
      --checkpoint "${TRAIN_OUT}/checkpoint_latest.pt" --out "${EVAL_OUT}" \
      --max-pairs "${TEST_PAIRS}" --max-new-tokens 24 --device cuda --dtype float16
  fi
}

gate() {
  "${PY}" "${SCRIPT_DIR}/check_native_reader_gate.py" \
    --eval "${EVAL_OUT}/SUCCESS.json" --out "${EVAL_OUT}" --paired-threshold 0.90
}

run_all() {
  audit
  cache_train
  cache_test
  train
  evaluate
  gate
}

wait_for_cuda() {
  printf '[%s] Waiting for CUDA before starting P2-G1.\n' "$(date -Is)"
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1; do
    printf '[%s] CUDA is unavailable; retrying in 60 seconds.\n' "$(date -Is)"
    sleep 60
  done
  printf '[%s] CUDA is available; starting P2-G1.\n' "$(date -Is)"
  run_all
}

status() {
  printf 'GPU:\n'
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  printf '\nProcesses:\n'
  pgrep -af 'cache_qwen3_4b_native_kv|train_qwen3_4b_native_reader|eval_qwen3_4b_native_reader' || true
  printf '\nMarkers:\n'
  find "${ROOT}/step1_native_reader" -maxdepth 3 -type f \
    \( -name 'AUDIT.json' -o -name 'CACHE_SUCCESS.json' -o -name 'TRAIN_SUCCESS.json' \
       -o -name 'SUCCESS.json' -o -name 'NATIVE_READER_GATE_*.json' \) -print 2>/dev/null || true
}

case "${1:-all}" in
  audit) audit ;;
  cache) audit; cache_train; cache_test ;;
  train) train ;;
  eval) evaluate ;;
  gate) gate ;;
  status) status ;;
  all) run_all ;;
  wait-cuda) wait_for_cuda ;;
  *) echo "Usage: $0 {audit|cache|train|eval|gate|status|all|wait-cuda}" >&2; exit 64 ;;
esac
