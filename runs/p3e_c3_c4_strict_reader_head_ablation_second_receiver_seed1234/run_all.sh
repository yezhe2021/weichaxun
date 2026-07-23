#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_c3_c4_strict_reader_head_ablation_second_receiver_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
P3EC1=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
P3EC2=/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
Q4=/home/yezhe/all_models/models/Qwen/Qwen3-4B
Q35=/home/yezhe/all_models/models/Qwen/Qwen3.5-4B
TRAIN_MEMORY=${P3D3}/cache/native/train/index.json
VALID_MEMORY=${P3D3}/cache/native/validation/index.json
WRITER=${P3EC2}/writer_formal512/worker/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${P3EC2}:${P3EC1}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() { until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done; }

audit() {
  mkdir -p "${ROOT}"/{audit,c3a,c3b,c4,logs}
  "${PY}" "${ROOT}/audit_c3_c4.py" --train-cache "${TRAIN_MEMORY}" --validation-cache "${VALID_MEMORY}" --writer "${WRITER}" --c2-success "${P3EC2}/SUCCESS.json" --qwen35 "${Q35}" --out "${ROOT}/audit/SUCCESS.json"
  wait_cuda
  "${PY}" "${ROOT}/smoke_c3_c4.py" --qwen4 "${Q4}" --qwen35 "${Q35}" --memory "${TRAIN_MEMORY}" --writer "${WRITER}" --device cuda
}

run_c3a_one() {
  local init=$1 seed=$2 name=$3
  wait_cuda
  "${PY}" "${ROOT}/train_c3a_reader.py" --model "${Q4}" --memory "${TRAIN_MEMORY}" --writer "${WRITER}" --out "${ROOT}/c3a/${name}/reader" --init "${init}" --seed "${seed}" --max-samples 512 --epochs "${C3A_EPOCHS:-20}" --device cuda
  "${PY}" "${ROOT}/eval_c3a_reader.py" --model "${Q4}" --memory "${VALID_MEMORY}" --writer "${WRITER}" --reader "${ROOT}/c3a/${name}/reader/checkpoint_best.pt" --out "${ROOT}/c3a/${name}/eval" --reader-role fresh_reader --max-samples 64 --seed "${seed}" --device cuda
}

c3a() {
  run_c3a_one fully_random 1234 random_seed1234
  run_c3a_one fully_random 2345 random_seed2345
  run_c3a_one weak_pair 1234 weak_pair_seed1234
  run_c3a_one weak_pair 2345 weak_pair_seed2345
  "${PY}" "${ROOT}/summarize_c3a.py" --random "${ROOT}/c3a/random_seed1234/eval/SUCCESS.json" "${ROOT}/c3a/random_seed2345/eval/SUCCESS.json" \
    --weak "${ROOT}/c3a/weak_pair_seed1234/eval/SUCCESS.json" "${ROOT}/c3a/weak_pair_seed2345/eval/SUCCESS.json" --out "${ROOT}/c3a/SUCCESS.json"
}

c3b() {
  wait_cuda
  "${PY}" "${ROOT}/eval_c3b_pair_ablation.py" --model "${Q4}" --memory "${VALID_MEMORY}" --writer "${WRITER}" --reader "${ROOT}/c3a/random_seed1234/reader/checkpoint_best.pt" --out "${ROOT}/c3b/random_seed1234" --max-samples 64 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_c3b_pair_ablation.py" --model "${Q4}" --memory "${VALID_MEMORY}" --writer "${WRITER}" --reader "${ROOT}/c3a/random_seed2345/reader/checkpoint_best.pt" --out "${ROOT}/c3b/random_seed2345" --max-samples 64 --seed 2345 --device cuda
  "${PY}" "${ROOT}/summarize_c3b.py" --runs "${ROOT}/c3b/random_seed1234/SUCCESS.json" "${ROOT}/c3b/random_seed2345/SUCCESS.json" --out "${ROOT}/c3b/SUCCESS.json"
}

run_c4_one() {
  local seed=$1 name=$2
  wait_cuda
  "${PY}" "${ROOT}/train_c4_qwen35_reader.py" --model "${Q35}" --memory "${TRAIN_MEMORY}" --writer "${WRITER}" --out "${ROOT}/c4/${name}/reader" --seed "${seed}" --max-samples 512 --epochs "${C4_EPOCHS:-20}" --device cuda
  "${PY}" "${ROOT}/eval_c4_qwen35_reader.py" --model "${Q35}" --memory "${VALID_MEMORY}" --writer "${WRITER}" --reader "${ROOT}/c4/${name}/reader/checkpoint_best.pt" --out "${ROOT}/c4/${name}/eval" --seed "${seed}" --max-samples 64 --device cuda
}

c4() {
  run_c4_one 1234 seed1234
  run_c4_one 2345 seed2345
  "${PY}" "${ROOT}/summarize_c4.py" --runs "${ROOT}/c4/seed1234/eval/SUCCESS.json" "${ROOT}/c4/seed2345/eval/SUCCESS.json" --out "${ROOT}/c4/SUCCESS.json"
  "${PY}" "${ROOT}/summarize_all.py" --c3a "${ROOT}/c3a/SUCCESS.json" --c3b "${ROOT}/c3b/SUCCESS.json" --c4 "${ROOT}/c4/SUCCESS.json" --writer "${WRITER}" --out "${ROOT}/SUCCESS.json"
}

case "${1:-all}" in
  audit) audit ;;
  c3a) c3a ;;
  c3b) c3b ;;
  c4) c4 ;;
  all) audit; c3a; c3b; c4 ;;
  *) echo "Usage: $0 {audit|c3a|c3b|c4|all}"; exit 2 ;;
esac
