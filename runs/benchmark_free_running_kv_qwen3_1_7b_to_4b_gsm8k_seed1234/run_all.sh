#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"

BASE_EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT:-/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
GSM8K_DATA="${GSM8K_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
METHODS="${METHODS:-native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional}"
CACHE_VARIANTS="${CACHE_VARIANTS:-correct,zero,shuffled,mismatched}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

case "${1:-help}" in
  run)
    "${PY}" "${ROOT}/run_benchmark_free_running_kv.py" \
      --base-experiment-root "${BASE_EXPERIMENT_ROOT}" \
      --sender-model "${SENDER}" \
      --receiver-model "${RECEIVER}" \
      --data "${GSM8K_DATA}" \
      --out "${ROOT}/results" \
      --methods "${METHODS}" \
      --cache-variants "${CACHE_VARIANTS}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-source-tokens "${MAX_SOURCE_TOKENS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  native)
    METHODS=native CACHE_VARIANTS=correct bash "$0" run
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash runs/benchmark_free_running_kv_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh native
  bash runs/benchmark_free_running_kv_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh run

Quick native check:
  MAX_SAMPLES=8 bash runs/benchmark_free_running_kv_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh native

Quick cache check:
  MAX_SAMPLES=8 METHODS=native,q_aware_functional CACHE_VARIANTS=correct,zero \
  bash runs/benchmark_free_running_kv_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh run

Defaults:
  Qwen3-1.7B -> Qwen3-4B
  local GSM8K
  native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional
  correct,zero,shuffled,mismatched
  MAX_SAMPLES=64
USAGE
    ;;
esac
