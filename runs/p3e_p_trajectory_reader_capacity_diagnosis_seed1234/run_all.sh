#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_p_trajectory_reader_capacity_diagnosis_seed1234
P3EN=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_MEMORY=${P3EN}/cache/train/index.json
VAL_MEMORY=${P3EN}/cache/validation/index.json

export PYTHONPATH="${ROOT}:${P3EN}:${P3EL}:${P3EA}:${P3D3}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}"/{teacher/train,teacher/validation,audit,train/a0,train/a1,train/c1,evaluation}

if [[ ! -f "${ROOT}/teacher/train/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_teacher_trajectory.py" \
    --model "${MODEL}" --memory "${TRAIN_MEMORY}" \
    --out "${ROOT}/teacher/train" --max-samples 512 --device cuda
fi
if [[ ! -f "${ROOT}/teacher/validation/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_teacher_trajectory.py" \
    --model "${MODEL}" --memory "${VAL_MEMORY}" \
    --out "${ROOT}/teacher/validation" --max-samples 64 --device cuda
fi

"${PY}" "${ROOT}/audit_teacher_student_alignment.py" \
  --model "${MODEL}" --memory "${TRAIN_MEMORY}" \
  --teacher-cache "${ROOT}/teacher/train/index.json" \
  --out "${ROOT}/audit" --max-samples 512

for MODE in a0 a1 c1; do
  "${PY}" "${ROOT}/train_trajectory_reader.py" \
    --model "${MODEL}" --memory "${TRAIN_MEMORY}" \
    --teacher-cache "${ROOT}/teacher/train/index.json" \
    --out "${ROOT}/train/${MODE}" --mode "${MODE}" \
    --max-samples 512 --epochs 8 --lr 1e-4 \
    --weight-decay 0.01 --seed 1234 --device cuda
done

"${PY}" "${ROOT}/eval_trajectory_readers.py" \
  --model "${MODEL}" --memory "${VAL_MEMORY}" \
  --teacher-cache "${ROOT}/teacher/validation/index.json" \
  --a0 "${ROOT}/train/a0/checkpoint_epoch_008.pt" \
  --a1 "${ROOT}/train/a1/checkpoint_epoch_008.pt" \
  --c1 "${ROOT}/train/c1/checkpoint_epoch_008.pt" \
  --out "${ROOT}/evaluation" --max-samples 64 \
  --max-new-tokens 32 --device cuda

"${PY}" "${ROOT}/build_manual_cpw_blind.py" \
  --records "${ROOT}/evaluation/per_sample_generation.jsonl" \
  --out "${ROOT}/evaluation/manual_cpw_blind.csv"

"${PY}" "${ROOT}/summarize.py" \
  --evaluation "${ROOT}/evaluation/SUCCESS.json" \
  --baseline "${P3EN}/evaluation/SUCCESS.json" \
  --audit "${ROOT}/audit/SUCCESS.json" \
  --out "${ROOT}/SUCCESS.json"

