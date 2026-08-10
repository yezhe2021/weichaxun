#!/bin/bash
set -e
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
cd /home/yezhe/不同模型的kv分析/scripts
WD=/home/yezhe/不同模型的kv分析/runs/qwen3_4b_8b_bidirectional_kv_scale_diagnosis_seed1234
CKPT=$WD/artifacts/development
M04=/home/yezhe/all_models/models/Qwen/Qwen3-4B
M08=/home/yezhe/all_models/models/Qwen/Qwen3-8B
run_direct() {
  local dir=$1 src=$2 tgt=$3 recv=$4 send=$5
  echo "[$(date +%H:%M:%S)] === direct $dir ==="
  $PY -u train_writer.py --workdir $WD --mode development --phase direct --direction $dir \
    --receiver-model $recv --source-dir $src --target-dir $tgt \
    --num-layers 36 --feature-dim 1024 --sampled-tokens 128 --max-updates 200 --max-new-tokens 32
  echo "[$(date +%H:%M:%S)] === evaluate $dir ==="
  $PY -u stage6_evaluate.py --workdir $WD --mode development \
    --receiver-model $recv --sender-model $send \
    --writer-checkpoint $CKPT/$dir/direct/best.pt --training-path f1_ce \
    --source-dir $src --target-dir $tgt \
    --num-layers 36 --feature-dim 1024 \
    --out $WD/evaluation/${dir}_fast_direct.json
}
run_direct self_04 source_04 target_04 $M04 $M04
run_direct self_08 source_08 target_08 $M08 $M08
run_direct 04_to_08 source_04 target_08 $M08 $M04
run_direct 08_to_04 source_08 target_04 $M04 $M08
echo '4B8B FAST DONE'
