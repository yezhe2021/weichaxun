#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
C1=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
C2=/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
VAL_BASE=${P3D3}/cache/native/validation/index.json
C1_READER=${C1}/formal512/reader/checkpoint_best.pt
C2_WRITER=${C2}/writer_formal512/worker/checkpoint_best.pt
NATIVE_READER=${P3EB}/formal512/reader/checkpoint_best.pt
CONTINUED=${ROOT}/conditioned_writer_epoch7_8

export PYTHONPATH=${ROOT}:${C2}:${C1}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PY}" "${ROOT}/train_conditioned_writer.py" --model "${RECEIVER}" --memory "${ROOT}/cache/train/index.json" \
  --init-writer "${ROOT}/conditioned_writer/checkpoint_best.pt" --native-reader "${NATIVE_READER}" \
  --canonical-reader "${C1_READER}" --out "${CONTINUED}" --max-samples 512 --epochs 2 --lr 1e-4 --seed 1240 --device cuda

"${PY}" "${ROOT}/eval_p3e_l.py" --model "${RECEIVER}" --base-memory "${VAL_BASE}" \
  --conditioned-memory "${ROOT}/cache/validation/index.json" --baseline-writer "${C2_WRITER}" \
  --conditioned-writer "${CONTINUED}/checkpoint_best.pt" --reader "${C1_READER}" \
  --out "${ROOT}/final_eval" --max-samples 64 --device cuda

"${PY}" "${ROOT}/summarize.py" --zero-shot "${ROOT}/zero_shot/SUCCESS.json" \
  --final "${ROOT}/final_eval/SUCCESS.json" --decision "${ROOT}/writer_decision.json" \
  --out "${ROOT}/SUCCESS.json"
