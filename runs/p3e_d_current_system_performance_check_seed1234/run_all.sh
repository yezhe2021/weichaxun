#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
P3EC1=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
P3EC2=/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
SENDER=/home/yezhe/all_models/models/Qwen/Qwen3-8B
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
MEMORY=${P3D3}/cache/native/validation/index.json
WRITER=${P3EC2}/writer_formal512/worker/checkpoint_best.pt
CANONICAL_READER=${P3EC1}/formal512/reader/checkpoint_best.pt
NATIVE_READER=${P3EB}/formal512/reader/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${P3EC2}:${P3EC1}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

prepare() {
  mkdir -p "${ROOT}"/{audit,sender_cache,evaluation,logs}
  "${PY}" "${ROOT}/prepare_manifest.py" \
    --memory "${MEMORY}" --writer "${WRITER}" --canonical-reader "${CANONICAL_READER}" --native-reader "${NATIVE_READER}" \
    --sender-model "${SENDER}" --receiver-model "${RECEIVER}" --out "${ROOT}" --max-samples 64 --seed 1234
  "${PY}" "${ROOT}/audit.py" --root "${ROOT}" --manifest "${ROOT}/manifest.json" --out "${ROOT}/audit/SUCCESS.json"
}

sender() {
  wait_cuda
  "${PY}" "${ROOT}/cache_sender_artifacts.py" \
    --model "${SENDER}" --memory "${MEMORY}" --writer "${WRITER}" --manifest "${ROOT}/manifest.json" \
    --out "${ROOT}/sender_cache" --summary-max-new-tokens 512 --device cuda
}

receiver() {
  wait_cuda
  "${PY}" "${ROOT}/eval_current_system.py" \
    --model "${RECEIVER}" --memory "${MEMORY}" --canonical-cache "${ROOT}/sender_cache/canonical" \
    --summaries "${ROOT}/sender_cache/sender_summaries.jsonl" --manifest "${ROOT}/manifest.json" \
    --native-reader "${NATIVE_READER}" --canonical-reader "${CANONICAL_READER}" \
    --out "${ROOT}/evaluation" --max-new-tokens 32 --seed 1234 --device cuda
}

summarize() {
  "${PY}" "${ROOT}/summarize.py" --manifest "${ROOT}/manifest.json" \
    --sender "${ROOT}/sender_cache/sender_timing.jsonl" --evaluation "${ROOT}/evaluation" --out "${ROOT}"
}

case "${1:-all}" in
  prepare) prepare ;;
  sender) sender ;;
  receiver) receiver ;;
  summarize) summarize ;;
  all) prepare; sender; receiver; summarize ;;
  *) echo "Usage: $0 {prepare|sender|receiver|summarize|all}"; exit 2 ;;
esac
