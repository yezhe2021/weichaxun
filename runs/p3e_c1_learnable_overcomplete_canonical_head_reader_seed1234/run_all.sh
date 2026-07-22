#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_MEMORY=${P3D3}/cache/native/train/index.json
VALIDATION_MEMORY=${P3D3}/cache/native/validation/index.json
NATIVE_READER=${P3EB}/formal512/reader/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

audit() {
  mkdir -p "${ROOT}"/{audit,overfit16,formal512,logs}
  "${PY}" "${ROOT}/audit_p3e_c1.py" --train-cache "${TRAIN_MEMORY}" --validation-cache "${VALIDATION_MEMORY}" \
    --native-reader "${NATIVE_READER}" --c0-success "${P3EC0}/SUCCESS.json" --out "${ROOT}/audit/SUCCESS.json"
  wait_cuda
  "${PY}" "${ROOT}/smoke_p3e_c1.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --native-reader "${NATIVE_READER}" --device cuda
}

overfit() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c1_reader.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --native-reader "${NATIVE_READER}" --out "${ROOT}/overfit16/reader" \
    --mode overfit16 --max-samples 16 --epochs "${P3E_C1_OVERFIT_EPOCHS:-30}" --rank 32 --gate-init 0.01 --top-k 2 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c1_reader.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --checkpoint "${ROOT}/overfit16/reader/checkpoint_best.pt" \
    --out "${ROOT}/overfit16/eval" --max-samples 16 --seed 1234 --device cuda
  "${PY}" "${ROOT}/check_overfit_gate.py" --result "${ROOT}/overfit16/eval/SUCCESS.json" --out "${ROOT}/overfit16/GATE.json"
}

formal() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c1_reader.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --native-reader "${NATIVE_READER}" --out "${ROOT}/formal512/reader" \
    --mode formal512 --max-samples 512 --epochs "${P3E_C1_FORMAL_EPOCHS:-20}" --rank 32 --gate-init 0.01 --top-k 2 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c1_reader.py" --model "${RECEIVER}" --memory "${VALIDATION_MEMORY}" --checkpoint "${ROOT}/formal512/reader/checkpoint_best.pt" \
    --out "${ROOT}/formal512/eval" --max-samples 64 --seed 1234 --device cuda
  "${PY}" "${ROOT}/summarize_p3e_c1.py" --c0 "${P3EC0}/SUCCESS.json" --c1 "${ROOT}/formal512/eval/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
}

case "${1:-all}" in
  audit) audit ;;
  overfit) overfit ;;
  formal) formal ;;
  all) audit; overfit; formal ;;
  *) echo "Usage: $0 {audit|overfit|formal|all}"; exit 2 ;;
esac
