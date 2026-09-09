#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_p_c2_uncompressed_multislot_trajectory_seed1234
P3EP=/home/yezhe/伪查询/runs/p3e_p_trajectory_reader_capacity_diagnosis_seed1234
P3EN=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B

export PYTHONPATH="${ROOT}:${P3EP}:${P3EN}:${P3EL}:${P3EA}:${P3D3}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}/train" "${ROOT}/evaluation"

"${PY}" "${ROOT}/train_c2.py" \
  --model "${MODEL}" \
  --memory "${P3EN}/cache/train/index.json" \
  --teacher-cache "${P3EP}/teacher/train/index.json" \
  --out "${ROOT}/train" --max-samples 512 \
  --epochs 8 --lr 1e-4 --weight-decay 0.01 \
  --seed 1234 --device cuda

"${PY}" "${ROOT}/eval_c2.py" \
  --model "${MODEL}" \
  --memory "${P3EN}/cache/validation/index.json" \
  --teacher-cache "${P3EP}/teacher/validation/index.json" \
  --checkpoint "${ROOT}/train/checkpoint_epoch_008.pt" \
  --out "${ROOT}/evaluation" --max-samples 64 \
  --max-new-tokens 32 --device cuda

"${PY}" "${ROOT}/build_manual_cpw_blind.py" \
  --records "${ROOT}/evaluation/per_sample_generation.jsonl" \
  --out "${ROOT}/evaluation/manual_cpw_blind.csv"

"${PY}" "${ROOT}/summarize_c2.py" \
  --p3ep "${P3EP}/SUCCESS.json" \
  --c2 "${ROOT}/evaluation/SUCCESS.json" \
  --out "${ROOT}/SUCCESS.json"

