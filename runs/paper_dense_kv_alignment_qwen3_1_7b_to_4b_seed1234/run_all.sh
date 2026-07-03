#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl}"
VAL_DATA="${VAL_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl}"
GSM8K_DATA="${GSM8K_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-64}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-1}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-1}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

train_one() {
  local method="$1"
  "${PY}" "${ROOT}/train_paper_dense_adapter.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/${method}" \
    --method "${method}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --phase1-epochs "${PHASE1_EPOCHS}" \
    --phase2-epochs "${PHASE2_EPOCHS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

checkpoint_for() {
  local method="$1"
  local checkpoint="${ROOT}/train/${method}/checkpoint_final.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  printf '%s\n' "${checkpoint}"
}

eval_with_data() {
  local method="$1"
  local data="$2"
  local out_dir="$3"
  "${PY}" "${ROOT}/eval_paper_dense_adapter.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${data}" \
    --adapter-checkpoint "$(checkpoint_for "${method}")" \
    --method-label "${method}" \
    --out "${out_dir}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --attention-topk "${ATTENTION_TOPK}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

token_diag_with_data() {
  local method="$1"
  local data="$2"
  local out_dir="$3"
  local critical_mode="$4"
  "${PY}" "${ROOT}/token_level_diagnostics.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${data}" \
    --adapter-checkpoint "$(checkpoint_for "${method}")" \
    --method-label "${method}" \
    --out "${out_dir}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --critical-mode "${critical_mode}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

eval_one() {
  eval_with_data "$1" "${VAL_DATA}" "${ROOT}/eval/$1"
}

eval_gsm8k_one() {
  eval_with_data "$1" "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/$1"
}

token_diag_hotpot_one() {
  token_diag_with_data "$1" "${VAL_DATA}" "${ROOT}/token_diag_hotpot/$1" answer
}

token_diag_gsm8k_one() {
  token_diag_with_data "$1" "${GSM8K_DATA}" "${ROOT}/token_diag_gsm8k/$1" numeric
}

case "${1:-help}" in
  train_paper) train_one paper_rec_then_mixed_generation ;;
  train_mse_only) train_one mse_only ;;
  train_mse_then_ce) train_one mse_then_ce ;;
  train_q_aware) train_one q_aware_functional ;;
  train)
    train_one mse_only
    train_one mse_then_ce
    train_one paper_rec_then_mixed_generation
    train_one q_aware_functional
    ;;
  eval_paper) eval_one paper_rec_then_mixed_generation ;;
  eval_mse_only) eval_one mse_only ;;
  eval_mse_then_ce) eval_one mse_then_ce ;;
  eval_q_aware) eval_one q_aware_functional ;;
  eval)
    eval_one mse_only
    eval_one mse_then_ce
    eval_one paper_rec_then_mixed_generation
    eval_one q_aware_functional
    ;;
  eval_gsm8k_paper) eval_gsm8k_one paper_rec_then_mixed_generation ;;
  eval_gsm8k_mse_only) eval_gsm8k_one mse_only ;;
  eval_gsm8k_mse_then_ce) eval_gsm8k_one mse_then_ce ;;
  eval_gsm8k_q_aware) eval_gsm8k_one q_aware_functional ;;
  eval_gsm8k)
    eval_gsm8k_one mse_only
    eval_gsm8k_one mse_then_ce
    eval_gsm8k_one paper_rec_then_mixed_generation
    eval_gsm8k_one q_aware_functional
    ;;
  token_diag_hotpot_paper) token_diag_hotpot_one paper_rec_then_mixed_generation ;;
  token_diag_hotpot_mse_only) token_diag_hotpot_one mse_only ;;
  token_diag_hotpot_mse_then_ce) token_diag_hotpot_one mse_then_ce ;;
  token_diag_hotpot_q_aware) token_diag_hotpot_one q_aware_functional ;;
  token_diag_hotpot)
    token_diag_hotpot_one mse_only
    token_diag_hotpot_one mse_then_ce
    token_diag_hotpot_one paper_rec_then_mixed_generation
    token_diag_hotpot_one q_aware_functional
    ;;
  token_diag_gsm8k_paper) token_diag_gsm8k_one paper_rec_then_mixed_generation ;;
  token_diag_gsm8k_mse_only) token_diag_gsm8k_one mse_only ;;
  token_diag_gsm8k_mse_then_ce) token_diag_gsm8k_one mse_then_ce ;;
  token_diag_gsm8k_q_aware) token_diag_gsm8k_one q_aware_functional ;;
  token_diag_gsm8k)
    token_diag_gsm8k_one mse_only
    token_diag_gsm8k_one mse_then_ce
    token_diag_gsm8k_one paper_rec_then_mixed_generation
    token_diag_gsm8k_one q_aware_functional
    ;;
  package) "${PY}" "${ROOT}/package_results.py" ;;
  package_gsm8k) "${PY}" "${ROOT}/package_gsm8k_results.py" ;;
  package_token_hotpot)
    "${PY}" "${ROOT}/package_token_diagnostics.py" \
      --dataset-label hotpotqa \
      --input-root "${ROOT}/token_diag_hotpot" \
      --out "${ROOT}/summary_token_diag/hotpotqa"
    ;;
  package_token_gsm8k)
    "${PY}" "${ROOT}/package_token_diagnostics.py" \
      --dataset-label gsm8k \
      --input-root "${ROOT}/token_diag_gsm8k" \
      --out "${ROOT}/summary_token_diag/gsm8k"
    ;;
  all)
    bash "$0" train
    bash "$0" eval
    bash "$0" package
    ;;
  *)
    cat <<'USAGE'
Usage:
  run_all.sh token_diag_hotpot
  run_all.sh token_diag_gsm8k
  run_all.sh token_diag_hotpot_q_aware
  run_all.sh token_diag_gsm8k_q_aware
  run_all.sh package_token_hotpot
  run_all.sh package_token_gsm8k

Existing entries are also preserved:
  train_paper|train_mse_only|train_mse_then_ce|train_q_aware|train
  eval_paper|eval_mse_only|eval_mse_then_ce|eval_q_aware|eval
  eval_gsm8k_paper|eval_gsm8k_mse_only|eval_gsm8k_mse_then_ce|eval_gsm8k_q_aware|eval_gsm8k
  package|package_gsm8k|all

Token diagnostics:
  HotpotQA critical tokens use critical-mode=answer.
  GSM8K critical tokens use critical-mode=numeric.

Environment overrides:
  MAX_VAL_SAMPLES=64
  MAX_SOURCE_TOKENS=256
  DEVICE=cuda|cpu
  DTYPE=float16|bfloat16|float32
  VAL_DATA=/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl
  GSM8K_DATA=/home/yezhe/数据集/gsm8k/test.jsonl
USAGE
    ;;
esac
