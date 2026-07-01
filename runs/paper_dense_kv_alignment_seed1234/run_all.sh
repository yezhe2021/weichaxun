#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
TRAIN_DATA="${TRAIN_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl}"
VAL_DATA="${VAL_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-64}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-1}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-1}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

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

eval_one() {
  local method="$1"
  local checkpoint="${ROOT}/train/${method}/checkpoint_final.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  "${PY}" "${ROOT}/eval_paper_dense_adapter.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${VAL_DATA}" \
    --adapter-checkpoint "${checkpoint}" \
    --method-label "${method}" \
    --out "${ROOT}/eval/${method}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --attention-topk "${ATTENTION_TOPK}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
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
  package) "${PY}" "${ROOT}/package_results.py" ;;
  all)
    bash "$0" train
    bash "$0" eval
    bash "$0" package
    ;;
  *)
    cat <<'USAGE'
Usage:
  run_all.sh train_paper
  run_all.sh train_mse_only
  run_all.sh train_mse_then_ce
  run_all.sh train_q_aware
  run_all.sh train
  run_all.sh eval_paper
  run_all.sh eval_mse_only
  run_all.sh eval_mse_then_ce
  run_all.sh eval_q_aware
  run_all.sh eval
  run_all.sh package
  run_all.sh all

Methods:
  paper_rec_then_mixed_generation: Phase I receiver-cache reconstruction + Phase II mixed context-aware/unaware generation loss
  mse_only: simple MSE baseline
  mse_then_ce: simple MSE then context-unaware gold CE baseline
  q_aware_functional: ours, same X protocol, Phase I reconstruction + mixed generation/logit-KL/Q-aware readout stage

Environment overrides:
  MAX_TRAIN_SAMPLES=512
  MAX_VAL_SAMPLES=64
  MAX_SOURCE_TOKENS=256
  PHASE1_EPOCHS=1
  PHASE2_EPOCHS=1
  DEVICE=cuda|cpu
  DTYPE=float16|float32
USAGE
    ;;
esac
