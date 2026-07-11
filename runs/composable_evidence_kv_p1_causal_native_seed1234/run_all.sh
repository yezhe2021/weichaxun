#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
FULL_TEXT_OUT="${FULL_TEXT_OUT:-${ROOT}/full_text_gate_qwen3_8b}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache_qwen3_8b}"
TRAIN_OUT="${TRAIN_OUT:-${ROOT}/train_qwen3_8b}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/eval_qwen3_8b}"
P15_TRAIN_OUT="${P15_TRAIN_OUT:-${ROOT}/p15_train_qwen3_8b}"
P15_EVAL_OUT="${P15_EVAL_OUT:-${ROOT}/p15_eval_qwen3_8b}"
BENCHMARK_OUT="${BENCHMARK_OUT:-${ROOT}/full_text_benchmark_qwen3_8b}"
IMPROVED_FULL_TEXT_OUT="${IMPROVED_FULL_TEXT_OUT:-${ROOT}/full_text_improved_qwen3_8b}"
BENCHMARK_SAMPLES="${BENCHMARK_SAMPLES:-256}"
SELF_CONSISTENCY_PATHS="${SELF_CONSISTENCY_PATHS:-5}"
TRAIN_PAIRS="${TRAIN_PAIRS:-2048}"
VALID_PAIRS="${VALID_PAIRS:-256}"
TEST_PAIRS="${TEST_PAIRS:-512}"
CACHE_MAX_SAMPLES="${CACHE_MAX_SAMPLES:-0}"
EPOCHS="${EPOCHS:-3}"
EVAL_PAIRS="${EVAL_PAIRS:-256}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

case "${1:-help}" in
  generate-data)
    "${PY}" "${ROOT}/generate_synthetic.py" \
      --out "${DATA_ROOT}" \
      --train-pairs "${TRAIN_PAIRS}" \
      --valid-pairs "${VALID_PAIRS}" \
      --test-pairs "${TEST_PAIRS}" \
      --seed "${SEED}"
    ;;
  full-text-gate)
    "${PY}" "${ROOT}/full_text_baseline.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${FULL_TEXT_OUT}" \
      --max-samples 256 \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  benchmark-all)
    "${PY}" "${ROOT}/benchmark_full_text.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${BENCHMARK_OUT}" \
      --max-samples "${BENCHMARK_SAMPLES}" \
      --self-consistency-paths "${SELF_CONSISTENCY_PATHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  full-text-improved)
    "${PY}" "${ROOT}/full_text_improved.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${IMPROVED_FULL_TEXT_OUT}" \
      --max-samples "${BENCHMARK_SAMPLES}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-train)
    "${PY}" "${ROOT}/cache_native_memory.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --out "${CACHE_ROOT}/train" \
      --max-samples "${CACHE_MAX_SAMPLES}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-test)
    "${PY}" "${ROOT}/cache_native_memory.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${CACHE_ROOT}/test" \
      --max-samples "${CACHE_MAX_SAMPLES}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  train)
    "${PY}" "${ROOT}/train_causal_adapter.py" \
      --model "${MODEL}" \
      --train-index "${CACHE_ROOT}/train/index.json" \
      --out "${TRAIN_OUT}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  eval)
    "${PY}" "${ROOT}/eval_causal_adapter.py" \
      --model "${MODEL}" \
      --test-index "${CACHE_ROOT}/test/index.json" \
      --checkpoint "${TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${EVAL_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  p15-train)
    "${PY}" "${ROOT}/train_p15_adapter.py" \
      --model "${MODEL}" \
      --train-index "${CACHE_ROOT}/train/index.json" \
      --out "${P15_TRAIN_OUT}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  p15-eval)
    "${PY}" "${ROOT}/eval_p15_adapter.py" \
      --model "${MODEL}" \
      --test-index "${CACHE_ROOT}/test/index.json" \
      --checkpoint "${P15_TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${P15_EVAL_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  p15-all)
    bash "${ROOT}/run_all.sh" cache-train
    bash "${ROOT}/run_all.sh" cache-test
    bash "${ROOT}/run_all.sh" p15-train
    bash "${ROOT}/run_all.sh" p15-eval
    ;;
  *)
    cat <<'USAGE'
Run each stage separately and inspect its output before continuing:

  bash run_all.sh generate-data
  bash run_all.sh full-text-gate
  bash run_all.sh benchmark-all
  bash run_all.sh cache-train
  bash run_all.sh cache-test
  bash run_all.sh train
  bash run_all.sh eval
  bash run_all.sh p15-train
  bash run_all.sh p15-eval
  bash run_all.sh p15-all

There is intentionally no all-in-one target.
USAGE
    ;;
esac
