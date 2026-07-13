#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache_native_kv_pairs}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/train}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT}/comparison}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
TEST_PAIRS="${TEST_PAIRS:-64}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

train_config() {
  local name="$1"
  local query_rank="$2"
  local output_rank="$3"
  "${PY}" "${ROOT}/train_p2a2_reader.py" \
    --model "${MODEL}" \
    --train-index "${CACHE_ROOT}/train/index.json" \
    --out "${TRAIN_ROOT}/${name}" \
    --config-name "${name}" \
    --query-rank "${query_rank}" \
    --output-rank "${output_rank}" \
    --max-pairs "${TRAIN_PAIRS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

eval_config() {
  local name="$1"
  "${PY}" "${ROOT}/eval_p2a2_reader.py" \
    --model "${MODEL}" \
    --test-index "${CACHE_ROOT}/test/index.json" \
    --checkpoint "${TRAIN_ROOT}/${name}/checkpoint_latest.pt" \
    --out "${EVAL_ROOT}/${name}" \
    --max-pairs "${EVAL_PAIRS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

case "${1:-help}" in
  generate-data)
    "${PY}" "${ROOT}/generate_counterfactual_data.py" \
      --model "${MODEL}" \
      --out "${DATA_ROOT}" \
      --train-pairs "${TRAIN_PAIRS}" \
      --test-pairs "${TEST_PAIRS}" \
      --seed "${SEED}"
    ;;
  cache-train)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --out "${CACHE_ROOT}/train" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-test)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${CACHE_ROOT}/test" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  train-output-only)
    train_config output_only 0 32
    ;;
  train-query-only)
    train_config query_only 32 0
    ;;
  train-query-output)
    train_config query_output 32 32
    ;;
  eval-output-only)
    eval_config output_only
    ;;
  eval-query-only)
    eval_config query_only
    ;;
  eval-query-output)
    eval_config query_output
    ;;
  summarize)
    "${PY}" "${ROOT}/summarize_p2a2.py" --root "${EVAL_ROOT}" --out "${SUMMARY_ROOT}"
    ;;
  all)
    bash "${ROOT}/run_all.sh" generate-data
    bash "${ROOT}/run_all.sh" cache-train
    bash "${ROOT}/run_all.sh" cache-test
    bash "${ROOT}/run_all.sh" train-output-only
    bash "${ROOT}/run_all.sh" eval-output-only
    bash "${ROOT}/run_all.sh" train-query-only
    bash "${ROOT}/run_all.sh" eval-query-only
    bash "${ROOT}/run_all.sh" train-query-output
    bash "${ROOT}/run_all.sh" eval-query-output
    bash "${ROOT}/run_all.sh" summarize
    ;;
  *)
    cat <<'USAGE'
P2-A2 complete run:
  bash run_all.sh all

Individual stages:
  bash run_all.sh generate-data
  bash run_all.sh cache-train
  bash run_all.sh cache-test
  bash run_all.sh train-output-only
  bash run_all.sh eval-output-only
  bash run_all.sh train-query-only
  bash run_all.sh eval-query-only
  bash run_all.sh train-query-output
  bash run_all.sh eval-query-output
  bash run_all.sh summarize
USAGE
    ;;
esac
