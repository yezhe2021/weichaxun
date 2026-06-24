#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/yezhe/伪查询
ROOT=${PROJECT}/runs/quantized_real_kv_translation_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
SENDER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B
RECEIVER=/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B
TRAIN_DATA=/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl
VAL_DATA=/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl

cd "${PROJECT}"

set_group() {
  case "$1" in
    fp16_fp16) SENDER_PRECISION=fp16; RECEIVER_PRECISION=fp16 ;;
    int4_fp16) SENDER_PRECISION=int4; RECEIVER_PRECISION=fp16 ;;
    fp16_int4) SENDER_PRECISION=fp16; RECEIVER_PRECISION=int4 ;;
    int4_int4) SENDER_PRECISION=int4; RECEIVER_PRECISION=int4 ;;
    *) echo "Unknown group: $1" >&2; exit 2 ;;
  esac
  GROUP="$1"
}

common_args() {
  printf '%s\n' \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --sender-precision "${SENDER_PRECISION}" \
    --receiver-precision "${RECEIVER_PRECISION}" \
    --dtype float16 \
    --device cuda \
    --seed 1234
}

preflight() {
  "${PY}" "${ROOT}/preflight.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --out "${ROOT}/preflight/report.json"
}

representation() {
  set_group "$1"
  mapfile -t COMMON < <(common_args)
  "${PY}" "${ROOT}/representation_probe.py" \
    "${COMMON[@]}" \
    --data "${VAL_DATA}" \
    --out "${ROOT}/representation/${GROUP}" \
    --max-samples 64 \
    --max-context-tokens 256 \
    --knn-k 8 \
    --attention-topk 16
}

drift() {
  local label="$1"
  local model
  case "${label}" in
    qwen3_0_6b) model="${SENDER}" ;;
    qwen3_1_7b) model="${RECEIVER}" ;;
    *) echo "Unknown drift model: ${label}" >&2; exit 2 ;;
  esac
  "${PY}" "${ROOT}/same_model_drift.py" \
    --model "${model}" \
    --model-label "${label}" \
    --data "${VAL_DATA}" \
    --out "${ROOT}/same_model_drift/${label}" \
    --max-samples 64 \
    --max-context-tokens 256 \
    --knn-k 8 \
    --dtype float16 \
    --device cuda \
    --seed 1234
}

train_mse_then_ce() {
  set_group "$1"
  mapfile -t COMMON < <(common_args)
  local base="${ROOT}/translation/${GROUP}/mse_then_ce"
  "${PY}" "${ROOT}/train_quantized_translator.py" \
    "${COMMON[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${base}/stage1_mse" \
    --objective mse \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --max-context-tokens 256 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512

  "${PY}" "${ROOT}/train_quantized_translator.py" \
    "${COMMON[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${base}/stage2_ce" \
    --objective ce \
    --init-checkpoint "${base}/stage1_mse/checkpoint_epoch1.pt" \
    --max-train-samples 512 \
    --max-val-samples 64 \
    --max-context-tokens 256 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --hidden 512
}

train_ce_only_small() {
  set_group "$1"
  mapfile -t COMMON < <(common_args)
  local base="${ROOT}/translation/${GROUP}/ce_only_small"
  "${PY}" "${ROOT}/train_quantized_translator.py" \
    "${COMMON[@]}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${base}/train" \
    --objective ce \
    --max-train-samples 256 \
    --max-val-samples 64 \
    --max-context-tokens 256 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --hidden 512
}

evaluate() {
  set_group "$1"
  local regime="$2"
  mapfile -t COMMON < <(common_args)
  local checkpoint
  case "${regime}" in
    mse_then_ce)
      checkpoint="${ROOT}/translation/${GROUP}/mse_then_ce/stage2_ce/checkpoint_epoch1.pt"
      ;;
    ce_only_small)
      checkpoint="${ROOT}/translation/${GROUP}/ce_only_small/train/checkpoint_epoch1.pt"
      ;;
    *) echo "Unknown regime: ${regime}" >&2; exit 2 ;;
  esac
  "${PY}" "${ROOT}/eval_quantized_translation.py" \
    "${COMMON[@]}" \
    --data "${VAL_DATA}" \
    --translator-checkpoint "${checkpoint}" \
    --out "${ROOT}/translation/${GROUP}/${regime}/eval" \
    --method-label "${GROUP}/${regime}/pure_translate" \
    --max-samples 64 \
    --max-context-tokens 256 \
    --max-new-tokens 32 \
    --attention-topk 16
}

usage() {
  cat <<'USAGE'
Usage:
  run_all.sh preflight
  run_all.sh representation GROUP
  run_all.sh drift {qwen3_0_6b|qwen3_1_7b}
  run_all.sh train_mse_then_ce GROUP
  run_all.sh train_ce_only_small GROUP
  run_all.sh eval GROUP {mse_then_ce|ce_only_small}
  run_all.sh package

GROUP is one of:
  fp16_fp16  int4_fp16  fp16_int4  int4_int4

Stages are intentionally explicit. This script does not launch every expensive job
from a single default command.
USAGE
}

case "${1:-help}" in
  preflight) preflight ;;
  representation) representation "${2:?GROUP is required}" ;;
  drift) drift "${2:?model label is required}" ;;
  train_mse_then_ce) train_mse_then_ce "${2:?GROUP is required}" ;;
  train_ce_only_small) train_ce_only_small "${2:?GROUP is required}" ;;
  eval) evaluate "${2:?GROUP is required}" "${3:?regime is required}" ;;
  package) "${PY}" "${ROOT}/package_results.py" ;;
  *) usage ;;
esac
