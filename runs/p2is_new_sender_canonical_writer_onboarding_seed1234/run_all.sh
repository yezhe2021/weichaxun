#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
P2IW_ROOT="${P2IW_ROOT:-${PROJECT}/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234}"
P2IR_ROOT="${P2IR_ROOT:-${PROJECT}/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
MODEL4="${MODEL4:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
MODEL35="${MODEL35:-/home/yezhe/all_models/models/Qwen/Qwen3___5-4B}"
Q8_TRAIN="${P2A2_ROOT}/cache_native_kv_pairs/train/index.json"
Q8_TEST="${P2A2_ROOT}/cache_native_kv_pairs/test/index.json"
OLD_TRAIN="${P2IR_ROOT}/cache/canonical/train/index.json"
OLD_TEST="${P2IR_ROOT}/cache/canonical/test/index.json"
READER4="${P2IR_ROOT}/qwen3_4b/full/train/checkpoint_best.pt"
READER35="${P2IR_ROOT}/qwen3_5_4b/full/train/checkpoint_best.pt"
P2IW_WRITER="${P2IW_ROOT}/writer_full/best_checkpoint.pt"
Q4_TRAIN="${ROOT}/cache/qwen3_4b_native/train/index.json"
Q4_TEST="${ROOT}/cache/qwen3_4b_native/test/index.json"
RIDGE="${ROOT}/ridge/q4_to_old_canonical.pt"
IMITATION="${ROOT}/imitation/train/checkpoint_best.pt"
TEACHER4="${ROOT}/cache/teacher/qwen3_4b/index.json"
TEACHER35="${ROOT}/cache/teacher/qwen3_5_4b/index.json"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"

export TOKENIZERS_PARALLELISM=false P2IW_ROOT P2IR_ROOT
cd "${PROJECT}"
wait_cuda() { until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done; }

audit() {
  "${PY}" -m py_compile "${ROOT}"/*.py
  "${PY}" "${ROOT}/audit_p2is.py" --p2iw-writer "${P2IW_WRITER}" --reader4 "${READER4}" --reader35 "${READER35}" \
    --old-train "${OLD_TRAIN}" --old-test "${OLD_TEST}" --q8-train "${Q8_TRAIN}" --q8-test "${Q8_TEST}" --model4 "${MODEL4}" --out "${ROOT}/AUDIT.json"
  wait_cuda; "${PY}" "${ROOT}/smoke_runtime.py" --device "${DEVICE}"
}

cache_native() {
  wait_cuda
  [[ -f "${Q4_TRAIN}" ]] || "${PY}" "${ROOT}/cache_qwen4_native.py" --model "${MODEL4}" --q8-native-index "${Q8_TRAIN}" --out "$(dirname "${Q4_TRAIN}")" --device "${DEVICE}" --dtype float16
  [[ -f "${Q4_TEST}" ]] || "${PY}" "${ROOT}/cache_qwen4_native.py" --model "${MODEL4}" --q8-native-index "${Q8_TEST}" --out "$(dirname "${Q4_TEST}")" --device "${DEVICE}" --dtype float16
}

fit_ridge() {
  [[ -f "${RIDGE}" ]] && return
  mkdir -p "$(dirname "${RIDGE}")"
  "${PY}" "${ROOT}/fit_ridge.py" --q4-index "${Q4_TRAIN}" --old-index "${OLD_TRAIN}" --out "${RIDGE}" --train-pairs 448 --ridge 10
}

imitation() {
  [[ -f "${ROOT}/imitation/train/TRAIN_SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/train_imitation.py" --q4-index "${Q4_TRAIN}" --old-index "${OLD_TRAIN}" --ridge "${RIDGE}" \
    --out "${ROOT}/imitation/train" --epochs 10 --rank 64 --lr 5e-4 --seed "${SEED}" --device "${DEVICE}"
}

teacher_cache() {
  local name="$1" model="$2" reader="$3" dtype="$4" out="${ROOT}/cache/teacher/$1"
  [[ -f "${out}/index.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/cache_teacher_logits.py" --receiver-name "${name}" --receiver-model "${model}" --reader-checkpoint "${reader}" \
    --old-index "${OLD_TRAIN}" --out "${out}" --topk 128 --device "${DEVICE}" --dtype "${dtype}"
}

cache_writer() {
  local checkpoint="$1" source_index="$2" out="$3" max_pairs="${4:-0}"
  [[ -f "${out}/index.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/cache_writer_memory.py" --q4-index "${source_index}" --ridge "${RIDGE}" --writer-checkpoint "${checkpoint}" --out "${out}" --max-pairs "${max_pairs}" --device "${DEVICE}"
}

evaluate() {
  local config="$1" new_index="$2" old_index="$3" max_pairs="$4" name="$5" model="$6" reader="$7" dtype="$8"
  local out="${ROOT}/evaluation/${config}/${name}"
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/eval_new_sender.py" --receiver-name "${name}" --receiver-model "${model}" --reader-checkpoint "${reader}" \
    --old-index "${old_index}" --new-index "${new_index}" --out "${out}" --max-pairs "${max_pairs}" --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

evaluate_both() {
  evaluate "$1" "$2" "$3" "$4" qwen3_4b "${MODEL4}" "${READER4}" float16
  evaluate "$1" "$2" "$3" "$4" qwen3_5_4b "${MODEL35}" "${READER35}" float32
}

calibrate() {
  local config="$1" train_pairs="$2" epochs="$3" out="$4"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/calibrate_writer.py" --config "${config}" --q4-index "${Q4_TRAIN}" --old-index "${OLD_TRAIN}" \
    --ridge "${RIDGE}" --init-writer "${IMITATION}" --teacher4 "${TEACHER4}" --teacher35 "${TEACHER35}" \
    --model4 "${MODEL4}" --model35 "${MODEL35}" --reader4 "${READER4}" --reader35 "${READER35}" \
    --out "${out}" --train-pairs "${train_pairs}" --epochs "${epochs}" --chunk-pairs 16 --lr 2e-4 --seed "${SEED}" --device "${DEVICE}"
}

run_all() {
  audit
  cache_native
  fit_ridge
  imitation
  teacher_cache qwen3_4b "${MODEL4}" "${READER4}" float16
  teacher_cache qwen3_5_4b "${MODEL35}" "${READER35}" float32

  cache_writer "${IMITATION}" "${Q4_TEST}" "${ROOT}/cache/writer/imitation_only/test" 64
  evaluate_both imitation_only "${ROOT}/cache/writer/imitation_only/test/index.json" "${OLD_TEST}" 64

  calibrate full 16 15 "${ROOT}/small_overfit/full"
  cache_writer "${ROOT}/small_overfit/full/checkpoint_best.pt" "${Q4_TRAIN}" "${ROOT}/cache/writer/small_full/train" 16
  evaluate_both small_full "${ROOT}/cache/writer/small_full/train/index.json" "${OLD_TRAIN}" 16

  calibrate q4_only 448 2 "${ROOT}/calibration/q4_only"
  calibrate dual_only 448 2 "${ROOT}/calibration/dual_only"
  calibrate full 448 3 "${ROOT}/calibration/full"

  for config in q4_only dual_only full; do
    cache_writer "${ROOT}/calibration/${config}/checkpoint_best.pt" "${Q4_TEST}" "${ROOT}/cache/writer/${config}/test" 64
    evaluate_both "${config}" "${ROOT}/cache/writer/${config}/test/index.json" "${OLD_TEST}" 64
  done
  cp "${ROOT}/calibration/full/checkpoint_best.pt" "${ROOT}/FINAL_W4B_CHECKPOINT.pt"
  "${PY}" "${ROOT}/summarize_p2is.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'cache_qwen4_native.py|fit_ridge.py|train_imitation.py|cache_teacher_logits.py|calibrate_writer.py|eval_new_sender.py|run_all.sh all' || true
  find "${ROOT}" -maxdepth 5 -type f \( -name 'AUDIT.json' -o -name 'TRAIN_SUCCESS.json' -o -name 'SUCCESS.json' -o -name 'CACHE_SUCCESS.json' \) -print 2>/dev/null | sort
}

case "${1:-help}" in
  audit) audit ;; cache-native) cache_native ;; ridge) fit_ridge ;; imitation) imitation ;;
  status) status ;; all) run_all ;;
  *) echo "Usage: bash run_all.sh {audit|cache-native|ridge|imitation|status|all}" >&2; exit 64 ;;
esac
