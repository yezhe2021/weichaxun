#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"

BASE_EXPERIMENT_ROOT="${BASE_EXPERIMENT_ROOT:-${PROJECT}/runs/paper_dense_kv_alignment_seed1234}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
VAL_DATA="${VAL_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl}"
GSM8K_DATA="${GSM8K_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
METHODS="${METHODS:-native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional}"
RECEIVER_PROMPT_MODES="${RECEIVER_PROMPT_MODES:-context_unaware,context_aware}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-64}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

cd "${PROJECT}"

run_dataset() {
  local dataset_label="$1"
  local data_path="$2"
  local critical_mode="$3"
  "${PY}" "${ROOT}/run_layerwise_logit_lens.py" \
    --base-experiment-root "${BASE_EXPERIMENT_ROOT}" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${data_path}" \
    --dataset-label "${dataset_label}" \
    --out "${ROOT}/${dataset_label}" \
    --methods "${METHODS}" \
    --receiver-prompt-modes "${RECEIVER_PROMPT_MODES}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --critical-mode "${critical_mode}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

plot_dataset() {
  local dataset_label="$1"
  "${PY}" "${ROOT}/plot_layerwise_logit_lens.py" \
    --summary "${ROOT}/${dataset_label}/layerwise_logit_lens_summary.csv" \
    --out "${ROOT}/${dataset_label}/plots"
}

case "${1:-help}" in
  gsm8k)
    run_dataset gsm8k "${GSM8K_DATA}" numeric
    plot_dataset gsm8k
    ;;
  hotpotqa)
    run_dataset hotpotqa "${VAL_DATA}" answer
    plot_dataset hotpotqa
    ;;
  package)
    "${PY}" "${ROOT}/package_layerwise_logit_lens.py" --root "${ROOT}" --out "${ROOT}/summary_all"
    ;;
  all)
    bash "$0" gsm8k
    bash "$0" hotpotqa
    bash "$0" package
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash runs/layerwise_logit_lens_diagnosis_seed1234/run_all.sh gsm8k
  bash runs/layerwise_logit_lens_diagnosis_seed1234/run_all.sh hotpotqa
  bash runs/layerwise_logit_lens_diagnosis_seed1234/run_all.sh package
  bash runs/layerwise_logit_lens_diagnosis_seed1234/run_all.sh all

Useful overrides:
  BASE_EXPERIMENT_ROOT=/home/yezhe/伪查询/runs/paper_dense_kv_alignment_seed1234
  BASE_EXPERIMENT_ROOT=/home/yezhe/伪查询/runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234
  SENDER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B
  RECEIVER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B
  DTYPE=float16
  DTYPE=bfloat16
  MAX_VAL_SAMPLES=64
  RECEIVER_PROMPT_MODES=context_unaware
  METHODS=native,mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional
USAGE
    ;;
esac
