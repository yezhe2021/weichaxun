#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EM=/home/yezhe/伪查询/runs/p3e_m_fresh_native_reader_q_conditioning_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_BASE=${P3D3}/cache/native/train/index.json
VAL_BASE=${P3D3}/cache/native/validation/index.json
TRAIN_MAP=${P3EL}/cache/train/index.json
VAL_MAP=${P3EL}/cache/validation/index.json
INIT=${P3EM}/shared_initialization.pt

export PYTHONPATH=${ROOT}:${P3EM}:${P3EL}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}"/{cache/train,cache/validation,audit,reader_c,evaluation}

if [[ ! -f "${ROOT}/cache/train/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_4b_qconditioned_native.py" --model "${MODEL}" \
    --base-memory "${TRAIN_BASE}" --mapping-index "${TRAIN_MAP}" \
    --out "${ROOT}/cache/train" --max-samples 512 --device cuda
fi
if [[ ! -f "${ROOT}/cache/validation/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_4b_qconditioned_native.py" --model "${MODEL}" \
    --base-memory "${VAL_BASE}" --mapping-index "${VAL_MAP}" \
    --out "${ROOT}/cache/validation" --max-samples 64 --device cuda
fi

"${PY}" "${ROOT}/audit.py" --train "${ROOT}/cache/train/index.json" \
  --validation "${ROOT}/cache/validation/index.json" --initialization "${INIT}" \
  --reader-b-training "${P3EM}/reader_b/TRAIN_SUCCESS.json" --out "${ROOT}/audit/SUCCESS.json"

"${PY}" "${ROOT}/train_reader_c.py" --model "${MODEL}" \
  --memory "${ROOT}/cache/train/index.json" --initialization "${INIT}" \
  --out "${ROOT}/reader_c" --max-samples 512 --epochs 5 --lr 2e-4 --seed 1234 --device cuda

"${PY}" "${ROOT}/eval_reader_c.py" --model "${MODEL}" \
  --memory "${ROOT}/cache/validation/index.json" \
  --checkpoint "${ROOT}/reader_c/checkpoint_epoch_005.pt" \
  --p3em-evaluation "${P3EM}/evaluation/SUCCESS.json" \
  --out "${ROOT}/evaluation" --max-samples 64 --max-new-tokens 32 --device cuda

"${PY}" "${ROOT}/summarize.py" --audit "${ROOT}/audit/SUCCESS.json" \
  --reader-c "${ROOT}/reader_c/TRAIN_SUCCESS.json" \
  --evaluation "${ROOT}/evaluation/SUCCESS.json" \
  --p3em "${P3EM}/evaluation/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
