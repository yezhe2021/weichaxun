#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_o_native_trajectory_oracle_patching_seed1234
P3EN=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B

export PYTHONPATH="${ROOT}:${P3EN}:${P3EL}:${P3EA}:${P3D3}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}/evaluation"

"${PY}" "${ROOT}/run_oracle_patching.py" \
  --model "${MODEL}" \
  --memory "${P3EN}/cache/validation/index.json" \
  --reader-checkpoint "${P3EN}/reader_c/checkpoint_epoch_005.pt" \
  --out "${ROOT}/evaluation" \
  --max-samples 64 --max-new-tokens 32 --device cuda

"${PY}" "${ROOT}/summarize.py" \
  --oracle "${ROOT}/evaluation/SUCCESS.json" \
  --baseline "${P3EN}/evaluation/SUCCESS.json" \
  --baseline-records "${P3EN}/evaluation/per_sample_generation.jsonl" \
  --out "${ROOT}/SUCCESS.json"

