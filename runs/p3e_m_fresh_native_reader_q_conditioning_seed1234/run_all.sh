#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_m_fresh_native_reader_q_conditioning_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_BASE=${P3D3}/cache/native/train/index.json
VAL_BASE=${P3D3}/cache/native/validation/index.json
TRAIN_Q=${P3EL}/cache/train/index.json
VAL_Q=${P3EL}/cache/validation/index.json
INIT=${ROOT}/shared_initialization.pt

export PYTHONPATH=${ROOT}:${P3EL}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}"/{audit,reader_a,reader_b,evaluation}

"${PY}" "${ROOT}/audit_inputs.py" --train-base "${TRAIN_BASE}" --train-conditioned "${TRAIN_Q}" \
  --validation-base "${VAL_BASE}" --validation-conditioned "${VAL_Q}" --out "${ROOT}/audit/SUCCESS.json"

"${PY}" "${ROOT}/make_shared_initialization.py" --model "${RECEIVER}" --out "${INIT}" --rank 32 --gate-init 0.01 --seed 1234 --device cuda

"${PY}" "${ROOT}/train_fresh_native_reader.py" --model "${RECEIVER}" --base-memory "${TRAIN_BASE}" \
  --conditioned-memory "${TRAIN_Q}" --initialization "${INIT}" --source evidence_only \
  --out "${ROOT}/reader_a" --max-samples 512 --epochs 5 --lr 2e-4 --seed 1234 --device cuda

"${PY}" "${ROOT}/train_fresh_native_reader.py" --model "${RECEIVER}" --base-memory "${TRAIN_BASE}" \
  --conditioned-memory "${TRAIN_Q}" --initialization "${INIT}" --source question_conditioned \
  --out "${ROOT}/reader_b" --max-samples 512 --epochs 5 --lr 2e-4 --seed 1234 --device cuda

"${PY}" "${ROOT}/eval_fresh_native_readers.py" --model "${RECEIVER}" --base-memory "${VAL_BASE}" \
  --conditioned-memory "${VAL_Q}" --reader-a "${ROOT}/reader_a/checkpoint_epoch_005.pt" \
  --reader-b "${ROOT}/reader_b/checkpoint_epoch_005.pt" --out "${ROOT}/evaluation" \
  --max-samples 64 --max-new-tokens 32 --device cuda

"${PY}" "${ROOT}/summarize.py" --audit "${ROOT}/audit/SUCCESS.json" \
  --reader-a "${ROOT}/reader_a/TRAIN_SUCCESS.json" --reader-b "${ROOT}/reader_b/TRAIN_SUCCESS.json" \
  --evaluation "${ROOT}/evaluation/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
