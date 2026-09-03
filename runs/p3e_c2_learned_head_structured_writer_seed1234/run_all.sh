#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
P3EC1=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_MEMORY=${P3D3}/cache/native/train/index.json
VALIDATION_MEMORY=${P3D3}/cache/native/validation/index.json
NATIVE_READER=${P3EB}/formal512/reader/checkpoint_best.pt
C1_READER=${P3EC1}/formal512/reader/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${P3EC1}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() { until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done; }

audit() {
  mkdir -p "${ROOT}"/{audit,writer_overfit16,writer_formal512,fresh_reader_overfit16,fresh_reader_formal512,logs}
  "${PY}" "${ROOT}/audit_p3e_c2.py" --train-cache "${TRAIN_MEMORY}" --validation-cache "${VALIDATION_MEMORY}" --native-reader "${NATIVE_READER}" --c1-reader "${C1_READER}" \
    --c1-success "${P3EC1}/SUCCESS.json" --c0-success "${P3EC0}/SUCCESS.json" --out "${ROOT}/audit/SUCCESS.json"
  wait_cuda
  "${PY}" "${ROOT}/smoke_p3e_c2.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --c1-reader "${C1_READER}" --device cuda
}

writer_overfit() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c2_writer.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --native-reader "${NATIVE_READER}" --canonical-reader "${C1_READER}" \
    --out "${ROOT}/writer_overfit16/worker" --mode overfit16 --max-samples 16 --epochs "${P3E_C2_WRITER_OVERFIT_EPOCHS:-20}" --rank 32 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c2.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --writer "${ROOT}/writer_overfit16/worker/checkpoint_best.pt" --reader "${C1_READER}" \
    --out "${ROOT}/writer_overfit16/eval" --reader-role writer_training_reader --max-samples 16 --seed 1234 --device cuda
  "${PY}" "${ROOT}/check_gate.py" --result "${ROOT}/writer_overfit16/eval/SUCCESS.json" --out "${ROOT}/writer_overfit16/GATE.json" || true
}

writer_formal() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c2_writer.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --native-reader "${NATIVE_READER}" --canonical-reader "${C1_READER}" \
    --out "${ROOT}/writer_formal512/worker" --mode formal512 --max-samples 512 --epochs "${P3E_C2_WRITER_FORMAL_EPOCHS:-15}" --rank 32 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c2.py" --model "${RECEIVER}" --memory "${VALIDATION_MEMORY}" --writer "${ROOT}/writer_formal512/worker/checkpoint_best.pt" --reader "${C1_READER}" \
    --out "${ROOT}/writer_formal512/eval" --reader-role writer_training_reader --max-samples 64 --seed 1234 --device cuda
}

fresh_overfit() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c2_fresh_reader.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --writer "${ROOT}/writer_formal512/worker/checkpoint_best.pt" --native-reader "${NATIVE_READER}" \
    --out "${ROOT}/fresh_reader_overfit16/reader" --mode overfit16 --max-samples 16 --epochs "${P3E_C2_FRESH_OVERFIT_EPOCHS:-30}" --seed 2345 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c2.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --writer "${ROOT}/writer_formal512/worker/checkpoint_best.pt" --reader "${ROOT}/fresh_reader_overfit16/reader/checkpoint_best.pt" \
    --out "${ROOT}/fresh_reader_overfit16/eval" --reader-role fresh_reader --max-samples 16 --seed 2345 --device cuda
  "${PY}" "${ROOT}/check_gate.py" --result "${ROOT}/fresh_reader_overfit16/eval/SUCCESS.json" --out "${ROOT}/fresh_reader_overfit16/GATE.json" || true
}

fresh_formal() {
  wait_cuda
  "${PY}" "${ROOT}/train_p3e_c2_fresh_reader.py" --model "${RECEIVER}" --memory "${TRAIN_MEMORY}" --writer "${ROOT}/writer_formal512/worker/checkpoint_best.pt" --native-reader "${NATIVE_READER}" \
    --out "${ROOT}/fresh_reader_formal512/reader" --mode formal512 --max-samples 512 --epochs "${P3E_C2_FRESH_FORMAL_EPOCHS:-20}" --seed 2345 --device cuda
  "${PY}" "${ROOT}/eval_p3e_c2.py" --model "${RECEIVER}" --memory "${VALIDATION_MEMORY}" --writer "${ROOT}/writer_formal512/worker/checkpoint_best.pt" --reader "${ROOT}/fresh_reader_formal512/reader/checkpoint_best.pt" \
    --out "${ROOT}/fresh_reader_formal512/eval" --reader-role fresh_reader --max-samples 64 --seed 2345 --device cuda
  "${PY}" "${ROOT}/summarize_p3e_c2.py" --native "${P3EB}/SUCCESS.json" --c0 "${P3EC0}/SUCCESS.json" --c1 "${P3EC1}/SUCCESS.json" \
    --writer-reader "${ROOT}/writer_formal512/eval/SUCCESS.json" --fresh-reader "${ROOT}/fresh_reader_formal512/eval/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
}

case "${1:-all}" in
  audit) audit ;;
  writer-overfit) writer_overfit ;;
  writer-formal) writer_formal ;;
  fresh-overfit) fresh_overfit ;;
  fresh-formal) fresh_formal ;;
  all) audit; writer_overfit; writer_formal; fresh_overfit; fresh_formal ;;
  *) echo "Usage: $0 {audit|writer-overfit|writer-formal|fresh-overfit|fresh-formal|all}"; exit 2 ;;
esac
