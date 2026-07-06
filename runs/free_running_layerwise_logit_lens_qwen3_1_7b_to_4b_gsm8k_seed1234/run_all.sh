#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"

BASE_EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT:-/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
GSM8K_DATA="${GSM8K_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
METHODS="${METHODS:-native,mse_only,paper_rec_then_mixed_generation,q_aware_functional}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

case "${1:-help}" in
  run)
    rm -f "${ROOT}/results/per_sample_trajectory.jsonl"
    "${PY}" "${ROOT}/run_free_running_layerwise_logit_lens.py" \
      --base-experiment-root "${BASE_EXPERIMENT_ROOT}" \
      --sender-model "${SENDER}" \
      --receiver-model "${RECEIVER}" \
      --data "${GSM8K_DATA}" \
      --out "${ROOT}/results" \
      --methods "${METHODS}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-source-tokens "${MAX_SOURCE_TOKENS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  plot)
    "${PY}" "${ROOT}/plot_free_running_layerwise.py" \
      --summary "${ROOT}/results/free_running_layerwise_summary.csv" \
      --out "${ROOT}/results/plots"
    ;;
  all)
    bash "$0" run
    bash "$0" plot
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash runs/free_running_layerwise_logit_lens_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh run
  bash runs/free_running_layerwise_logit_lens_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh plot
  bash runs/free_running_layerwise_logit_lens_qwen3_1_7b_to_4b_gsm8k_seed1234/run_all.sh all

Defaults:
  Qwen3-1.7B -> Qwen3-4B
  GSM8K
  context_unaware
  native,mse_only,paper_rec_then_mixed_generation,q_aware_functional
  MAX_SAMPLES=64
  MAX_NEW_TOKENS=96
  DTYPE=bfloat16

Useful overrides:
  MAX_SAMPLES=16
  MAX_NEW_TOKENS=64
  METHODS=native,q_aware_functional
  DEVICE=cuda
USAGE
    ;;
esac
