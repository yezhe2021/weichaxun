#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/yezhe/伪查询
ROOT=${PROJECT}/runs/real_qwen3_0_6b_to_1_7b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
SENDER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B
RECEIVER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B
TRAIN_DATA=/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl
VAL_DATA=/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl

cd "${PROJECT}"

COMMON_ARGS=(
  --sender-model "${SENDER}"
  --receiver-model "${RECEIVER}"
  --max-context-tokens 256
  --seed 1234
  --device cuda
  --dtype float16
)

train_mse_only() {
  "${PY}" "${ROOT}/train_real_kv_translator.py" \
    "${COMMON_ARGS[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/mse_only" \
    --objective mse \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512
}

train_ce_only() {
  "${PY}" "${ROOT}/train_real_kv_translator.py" \
    "${COMMON_ARGS[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/ce_only" \
    --objective ce \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512
}

train_mse_ce() {
  "${PY}" "${ROOT}/train_real_kv_translator.py" \
    "${COMMON_ARGS[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/mse_ce" \
    --objective mse_ce \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512
}

train_mse_then_ce() {
  "${PY}" "${ROOT}/train_real_kv_translator.py" \
    "${COMMON_ARGS[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/mse_then_ce_stage1" \
    --objective mse \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512

  "${PY}" "${ROOT}/train_real_kv_translator.py" \
    "${COMMON_ARGS[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/mse_then_ce" \
    --objective ce \
    --init-checkpoint "${ROOT}/train/mse_then_ce_stage1/checkpoint_epoch1.pt" \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --hidden 512
}

eval_one() {
  local name="$1"
  "${PY}" "${ROOT}/eval_real_kv_translation.py" \
    "${COMMON_ARGS[@]}" \
    --data "${VAL_DATA}" \
    --translator-checkpoint "${ROOT}/train/${name}/checkpoint_epoch1.pt" \
    --out "${ROOT}/eval/${name}" \
    --method-label "pure_translate/${name}" \
    --max-samples 64 \
    --attention-topk 16
}

case "${1:-help}" in
  train_mse_only) train_mse_only ;;
  train_ce_only) train_ce_only ;;
  train_mse_ce) train_mse_ce ;;
  train_mse_then_ce) train_mse_then_ce ;;
  train)
    train_mse_only
    train_ce_only
    train_mse_ce
    train_mse_then_ce
    ;;
  eval_mse_only) eval_one mse_only ;;
  eval_ce_only) eval_one ce_only ;;
  eval_mse_ce) eval_one mse_ce ;;
  eval_mse_then_ce) eval_one mse_then_ce ;;
  eval)
    eval_one mse_only
    eval_one ce_only
    eval_one mse_ce
    eval_one mse_then_ce
    ;;
  package)
    "${PY}" "${ROOT}/package_results.py"
    ;;
  all)
    "$0" train
    "$0" eval
    "$0" package
    ;;
  *)
    cat <<USAGE
Usage: $0 {train_mse_only|train_ce_only|train_mse_ce|train_mse_then_ce|train|eval_mse_only|eval_ce_only|eval_mse_ce|eval_mse_then_ce|eval|package|all}

This real cross-model experiment translates context C KV only:
  Qwen3-0.6B prefill(C) -> translator -> Qwen3-1.7B-shaped context KV
  Qwen3-1.7B then natively processes Q + answer teacher forcing.
USAGE
    ;;
esac
