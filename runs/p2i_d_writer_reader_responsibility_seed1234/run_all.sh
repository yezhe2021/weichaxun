#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
P2I_ROOT="${P2I_ROOT:-${PROJECT}/runs/p2i_cached_canonical_evidence_kv_qwen3_8b_seed1234}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
RECEIVER4_MODEL="${RECEIVER4_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
RECEIVER35_MODEL="${RECEIVER35_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3___5-4B}"
CANONICAL_TRAIN="${CANONICAL_TRAIN:-${P2I_ROOT}/cache/canonical/train/index.json}"
CANONICAL_TEST="${CANONICAL_TEST:-${P2I_ROOT}/cache/canonical/test/index.json}"
NATIVE_TRAIN="${NATIVE_TRAIN:-${P2A2_ROOT}/cache_native_kv_pairs/train/index.json}"
NATIVE_TEST="${NATIVE_TEST:-${P2A2_ROOT}/cache_native_kv_pairs/test/index.json}"
MOTHER_CHECKPOINT_FILE="${P2I_ROOT}/mother/FINAL_CHECKPOINT.txt"
SUBSET="${ROOT}/data/diagnostic_subset.json"
PROBE="${ROOT}/writer_probe"
SLOT_DIAG="${ROOT}/slot_diagnostics"
READER4="${ROOT}/reader_oracle/qwen3_4b"
READER35="${ROOT}/reader_oracle/qwen3_5_4b"
JOINT="${ROOT}/joint_overfit/direct_qwen3_4b"
JOINT_RESCUE="${ROOT}/joint_overfit/reader_warmup_qwen3_4b"
RESPONSIBILITY="${ROOT}/responsibility"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"

export TOKENIZERS_PARALLELISM=false
export P2I_ROOT
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { printf 'Required file is missing: %s\n' "$1" >&2; exit 1; }
}

mother_checkpoint() {
  require_file "${MOTHER_CHECKPOINT_FILE}"
  local checkpoint
  checkpoint="$(cat "${MOTHER_CHECKPOINT_FILE}")"
  require_file "${checkpoint}"
  printf '%s' "${checkpoint}"
}

audit() {
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" -m py_compile "${ROOT}"/*.py
  "${PY}" "${ROOT}/audit_p2id.py" \
    --p2i-root "${P2I_ROOT}" --mother-checkpoint "${checkpoint}" \
    --canonical-train-index "${CANONICAL_TRAIN}" --canonical-test-index "${CANONICAL_TEST}" \
    --native-train-index "${NATIVE_TRAIN}" --native-test-index "${NATIVE_TEST}" \
    --receiver4-model "${RECEIVER4_MODEL}" --receiver35-model "${RECEIVER35_MODEL}" \
    --out "${ROOT}/AUDIT.json"
}

make_subset() {
  [[ -f "${SUBSET}" ]] && return
  mkdir -p "${ROOT}/data"
  "${PY}" "${ROOT}/make_subset.py" \
    --index "${NATIVE_TRAIN}" --out "${SUBSET}" --pairs 8 --seed "${SEED}"
}

writer_probe() {
  [[ -f "${PROBE}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/writer_probe.py" \
    --train-index "${CANONICAL_TRAIN}" --test-index "${CANONICAL_TEST}" \
    --out "${PROBE}" --train-pairs 448 --validation-pairs 64 \
    --epochs 30 --patience 5 --seed "${SEED}" --device "${DEVICE}"
}

slot_diagnostics() {
  [[ -f "${SLOT_DIAG}/SUCCESS.json" ]] && return
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" "${ROOT}/slot_diagnostics.py" \
    --writer-checkpoint "${checkpoint}" --native-train-index "${NATIVE_TRAIN}" \
    --native-test-index "${NATIVE_TEST}" --out "${SLOT_DIAG}" \
    --train-pairs 448 --validation-pairs 64 --device "${DEVICE}" --dtype float16
}

reader_oracle() {
  local receiver_name="$1" model="$2" dtype="$3" out="$4"
  [[ -f "${out}/SUCCESS.json" ]] && return
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" "${ROOT}/reader_free_slots_oracle.py" \
    --receiver-name "${receiver_name}" --receiver-model "${model}" \
    --mother-checkpoint "${checkpoint}" --canonical-index "${CANONICAL_TRAIN}" \
    --subset "${SUBSET}" --out "${out}" --reader-init mother \
    --epochs 60 --patience 15 --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

reader_oracles() {
  make_subset
  reader_oracle qwen3_4b "${RECEIVER4_MODEL}" float16 "${READER4}"
  reader_oracle qwen3_5_4b "${RECEIVER35_MODEL}" float32 "${READER35}"
}

joint_gate() {
  "${PY}" "${ROOT}/responsibility.py" --mode joint-gate \
    --writer-probe "${PROBE}/SUCCESS.json" \
    --reader4 "${READER4}/SUCCESS.json" --reader35 "${READER35}/SUCCESS.json" \
    --out "${RESPONSIBILITY}"
}

joint_run() {
  local out="$1" warmup="$2"
  [[ -f "${out}/SUCCESS.json" ]] && return
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" "${ROOT}/joint_overfit.py" \
    --receiver-name qwen3_4b --receiver-model "${RECEIVER4_MODEL}" \
    --mother-checkpoint "${checkpoint}" --native-index "${NATIVE_TRAIN}" \
    --subset "${SUBSET}" --out "${out}" --epochs 60 --patience 15 \
    --reader-warmup-epochs "${warmup}" --seed "${SEED}" --device "${DEVICE}" --dtype float16
}

joint_overfit() {
  if joint_gate; then
    joint_run "${JOINT}" 0
    if ! "${PY}" -c "import json; raise SystemExit(0 if json.load(open('${JOINT}/SUCCESS.json'))['joint_overfit_passed'] else 1)"; then
      joint_run "${JOINT_RESCUE}" 10
    fi
  else
    printf 'Joint overfit skipped: a positive-control prerequisite failed.\n'
  fi
}

summarize() {
  local joint_args=()
  [[ -f "${JOINT}/SUCCESS.json" ]] && joint_args+=(--joint "${JOINT}/SUCCESS.json")
  [[ -f "${JOINT_RESCUE}/SUCCESS.json" ]] && joint_args+=(--joint-rescue "${JOINT_RESCUE}/SUCCESS.json")
  "${PY}" "${ROOT}/responsibility.py" --mode summarize \
    --writer-probe "${PROBE}/SUCCESS.json" \
    --reader4 "${READER4}/SUCCESS.json" --reader35 "${READER35}/SUCCESS.json" \
    --out "${RESPONSIBILITY}" "${joint_args[@]}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'writer_probe.py|slot_diagnostics.py|reader_free_slots_oracle.py|joint_overfit.py' || true
  find "${ROOT}" -maxdepth 4 -type f \
    \( -name 'AUDIT.json' -o -name 'SUCCESS.json' -o -name 'JOINT_VALIDITY_GATE.json' \) \
    -print 2>/dev/null | sort
}

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do
    sleep 60
  done
  "$0" all
}

case "${1:-help}" in
  audit) audit ;;
  subset) make_subset ;;
  probe) writer_probe ;;
  slot-diagnostics) slot_diagnostics ;;
  reader-oracles) reader_oracles ;;
  joint) joint_overfit ;;
  summarize) summarize ;;
  status) status ;;
  wait-cuda) wait_cuda ;;
  all)
    audit
    make_subset
    writer_probe
    slot_diagnostics
    reader_oracles
    joint_overfit
    summarize
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|subset|probe|slot-diagnostics|reader-oracles|joint|summarize|status|wait-cuda|all}" >&2
    exit 64
    ;;
esac
