#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
SENDER=/home/yezhe/all_models/models/Qwen/Qwen3-8B
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_MEMORY=${P3D3}/cache/native/train/index.json
VALIDATION_MEMORY=${P3D3}/cache/native/validation/index.json
STAGE_A_CHECKPOINT=${P3EA}/formal512/reader/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

audit() {
  mkdir -p "${ROOT}"/{audit,zero_shot,overfit16,formal512,logs}
  "${PY}" "${ROOT}/audit_p3e_b.py" --sender "${SENDER}" --receiver "${RECEIVER}" --train-cache "${TRAIN_MEMORY}" --validation-cache "${VALIDATION_MEMORY}" \
    --train-data "${P3D3}/data/train.jsonl" --validation-data "${P3D3}/data/validation.jsonl" --stage-a-gate "${P3EA}/overfit16/GATE.json" --out "${ROOT}/audit/SUCCESS.json"
  wait_cuda
  "${PY}" "${ROOT}/smoke_p3e_b.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --stage-a-checkpoint "${STAGE_A_CHECKPOINT}" --device cuda
}

zero_shot() {
  wait_cuda
  "${PY}" "${ROOT}/eval_sender_native_headwise.py" --model "${RECEIVER}" --memory "${VALIDATION_MEMORY}" --checkpoint "${STAGE_A_CHECKPOINT}" \
    --out "${ROOT}/zero_shot/eval" --reader-mode stage_a_zero_shot --max-samples 64 --seed 1234 --device cuda
}

overfit() {
  wait_cuda
  "${PY}" "${ROOT}/train_sender_native_headwise.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --out "${ROOT}/overfit16/reader" \
    --mode overfit16 --max-samples 16 --epochs "${P3E_B_OVERFIT_EPOCHS:-30}" --rank 32 --gate-init 0.01 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_sender_native_headwise.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --checkpoint "${ROOT}/overfit16/reader/checkpoint_best.pt" \
    --out "${ROOT}/overfit16/eval" --reader-mode stage_b_retrained --max-samples 16 --seed 1234 --device cuda
  "${PY}" "${ROOT}/check_overfit_gate.py" --result "${ROOT}/overfit16/eval/SUCCESS.json" --out "${ROOT}/overfit16/GATE.json"
}

formal() {
  wait_cuda
  "${PY}" "${ROOT}/train_sender_native_headwise.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --out "${ROOT}/formal512/reader" \
    --mode formal512 --max-samples 512 --epochs "${P3E_B_FORMAL_EPOCHS:-20}" --rank 32 --gate-init 0.01 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_sender_native_headwise.py" --model "${RECEIVER}" --memory "${VALIDATION_MEMORY}" --checkpoint "${ROOT}/formal512/reader/checkpoint_best.pt" \
    --out "${ROOT}/formal512/eval" --reader-mode stage_b_retrained --max-samples 64 --seed 1234 --device cuda
  "${PY}" "${ROOT}/summarize_p3e_b.py" --stage-a "${P3EA}/formal512/eval/SUCCESS.json" --zero-shot "${ROOT}/zero_shot/eval/SUCCESS.json" \
    --retrained "${ROOT}/formal512/eval/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
}

case "${1:-all}" in
  audit) audit ;;
  zero-shot) zero_shot ;;
  overfit) overfit ;;
  formal) formal ;;
  all) audit; zero_shot; overfit; formal ;;
  *) echo "Usage: $0 {audit|zero-shot|overfit|formal|all}"; exit 2 ;;
esac
