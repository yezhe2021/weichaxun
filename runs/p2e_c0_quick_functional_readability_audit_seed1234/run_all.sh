#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
P2A2="${P2A2:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2E="${P2E:-${PROJECT}/runs/p2e_llama3_2_3b_to_qwen3_8b_writer_seed1234}"
P2EA="${P2EA:-${PROJECT}/runs/p2e_a_prefill_state_answerability_llama3_2_3b_seed1234}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
RAW_CACHE="${RAW_CACHE:-${P2E}/cache_llama3_2_3b_native_kv_pairs}"
HIDDEN_CACHE="${HIDDEN_CACHE:-${P2EA}/cache_prefill_states}"
NATIVE_CACHE="${NATIVE_CACHE:-${P2A2}/cache_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2}/train/query_only/checkpoint_latest.pt}"
TASK_WRITER="${TASK_WRITER:-${P2E}/train/task_only/checkpoint_latest.pt}"
SHARED_WRITER="${SHARED_WRITER:-${P2E}/train/shared_span_relation/checkpoint_latest.pt}"
MANIFEST="${MANIFEST:-${ROOT}/split_manifest.json}"
MEMORY_RESULTS="${MEMORY_RESULTS:-${ROOT}/memory_probes}"
CHAIN_CACHE="${CHAIN_CACHE:-${ROOT}/reader_chain_cache}"
CHAIN_RESULTS="${CHAIN_RESULTS:-${ROOT}/chain_probes}"
GEN_RESULTS="${GEN_RESULTS:-${ROOT}/generation}"
COMPARISON="${COMPARISON:-${ROOT}/comparison}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-1234}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() { [[ -f "$1" ]] || { echo "Missing: $1" >&2; exit 1; }; }

audit() {
  for path in \
    "${RAW_CACHE}/train/index.json" "${RAW_CACHE}/test/index.json" \
    "${HIDDEN_CACHE}/train/index.json" "${HIDDEN_CACHE}/test/index.json" \
    "${NATIVE_CACHE}/test/index.json" "${READER_CHECKPOINT}" \
    "${TASK_WRITER}" "${SHARED_WRITER}"; do require_file "${path}"; done
  "${PY}" -m py_compile \
    "${ROOT}/audit_common.py" "${ROOT}/build_quick_manifest.py" \
    "${ROOT}/functional_probes.py" "${ROOT}/train_memory_probes.py" \
    "${ROOT}/cache_reader_chain.py" "${ROOT}/train_chain_probes.py" \
    "${ROOT}/eval_quick_generation.py" "${ROOT}/summarize_quick_audit.py" \
    "${ROOT}/smoke_quick_audit.py"
  "${PY}" "${ROOT}/smoke_quick_audit.py"
}

manifest() {
  [[ -f "${MANIFEST}" ]] && return
  "${PY}" "${ROOT}/build_quick_manifest.py" \
    --train-index "${RAW_CACHE}/train/index.json" \
    --test-index "${RAW_CACHE}/test/index.json" \
    --out "${MANIFEST}" --train-pairs 64 --validation-pairs 16 --test-pairs 16 --seed "${SEED}"
}

memory_probes() {
  [[ -f "${MEMORY_RESULTS}/MEMORY_STAGES_SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/train_memory_probes.py" \
    --manifest "${MANIFEST}" \
    --raw-train-index "${RAW_CACHE}/train/index.json" --raw-test-index "${RAW_CACHE}/test/index.json" \
    --hidden-train-index "${HIDDEN_CACHE}/train/index.json" --hidden-test-index "${HIDDEN_CACHE}/test/index.json" \
    --task-writer-checkpoint "${TASK_WRITER}" --shared-writer-checkpoint "${SHARED_WRITER}" \
    --out "${MEMORY_RESULTS}" --epochs 6 --seed "${SEED}" --device "${DEVICE}"
}

chain_branch() {
  local name="$1" writer="$2"
  local cache="${CHAIN_CACHE}/${name}" result="${CHAIN_RESULTS}/${name}"
  if [[ ! -f "${cache}/CACHE_SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/cache_reader_chain.py" \
      --manifest "${MANIFEST}" --raw-train-index "${RAW_CACHE}/train/index.json" \
      --raw-test-index "${RAW_CACHE}/test/index.json" --writer-checkpoint "${writer}" \
      --reader-checkpoint "${READER_CHECKPOINT}" --receiver-model "${RECEIVER_MODEL}" \
      --out "${cache}" --device "${DEVICE}" --dtype "${DTYPE}"
  fi
  if [[ ! -f "${result}/CHAIN_STAGES_SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/train_chain_probes.py" \
      --manifest "${MANIFEST}" --chain-cache "${cache}" --out "${result}" \
      --epochs 6 --seed "${SEED}" --device "${DEVICE}"
  fi
}

generation_branch() {
  local name="$1" writer="$2" out="${GEN_RESULTS}/$1"
  [[ -f "${out}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/eval_quick_generation.py" \
    --manifest "${MANIFEST}" --raw-test-index "${RAW_CACHE}/test/index.json" \
    --native-test-index "${NATIVE_CACHE}/test/index.json" --writer-checkpoint "${writer}" \
    --reader-checkpoint "${READER_CHECKPOINT}" --receiver-model "${RECEIVER_MODEL}" \
    --out "${out}" --max-new-tokens 24 --device "${DEVICE}" --dtype "${DTYPE}"
}

summarize() {
  "${PY}" "${ROOT}/summarize_quick_audit.py" \
    --memory-root "${MEMORY_RESULTS}" \
    --task-chain-root "${CHAIN_RESULTS}/task_only" \
    --shared-chain-root "${CHAIN_RESULTS}/shared_span_relation" \
    --task-generation "${GEN_RESULTS}/task_only" \
    --shared-generation "${GEN_RESULTS}/shared_span_relation" \
    --out "${COMPARISON}"
}

case "${1:-help}" in
  audit) audit ;;
  manifest) manifest ;;
  memory) memory_probes ;;
  chain-task) chain_branch task_only "${TASK_WRITER}" ;;
  chain-shared) chain_branch shared_span_relation "${SHARED_WRITER}" ;;
  generation-task) generation_branch task_only "${TASK_WRITER}" ;;
  generation-shared) generation_branch shared_span_relation "${SHARED_WRITER}" ;;
  summarize) summarize ;;
  all)
    audit; manifest; memory_probes
    chain_branch task_only "${TASK_WRITER}"
    chain_branch shared_span_relation "${SHARED_WRITER}"
    generation_branch task_only "${TASK_WRITER}"
    generation_branch shared_span_relation "${SHARED_WRITER}"
    summarize
    ;;
  status)
    [[ -f "${MANIFEST}" ]] && echo manifest=complete || echo manifest=pending
    [[ -f "${MEMORY_RESULTS}/MEMORY_STAGES_SUCCESS.json" ]] && echo memory=complete || echo memory=pending
    for name in task_only shared_span_relation; do
      [[ -f "${CHAIN_RESULTS}/${name}/CHAIN_STAGES_SUCCESS.json" ]] && chain=complete || chain=pending
      [[ -f "${GEN_RESULTS}/${name}/SUCCESS.json" ]] && generation=complete || generation=pending
      echo "${name}: chain=${chain} generation=${generation}"
    done
    [[ -f "${COMPARISON}/SUCCESS.json" ]] && echo comparison=complete || echo comparison=pending
    ;;
  *) echo "Usage: bash run_all.sh {audit|manifest|memory|chain-task|chain-shared|generation-task|generation-shared|summarize|all|status}" ;;
esac
