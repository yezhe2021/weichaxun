# Reverse Full-36 Writer: Qwen3-4B to Qwen3-1.7B

This is the direction-reversed counterpart of the Full-28 experiment. The
Sender is Qwen3-4B, the frozen Receiver is Qwen3-1.7B, and all other protocol
choices remain locked.

## Writer

```text
Qwen3-4B Native Full KV [36,T,8,128]
  -> concatenate source layer only [T,8,4608]
  -> 28 independent K Linear(4608,128)
  -> 28 independent V Linear(4608,128)
  -> predicted Qwen3-1.7B Native KV [28,T,8,128]
```

K/V parameters and target layers are independent. No token or head mixing is
allowed. Every linear has `bias=False`; therefore `Writer(0)=0`.

The reverse Full-36 Writer has exactly 33,030,144 parameters, equal to the
forward Full-28 Writer.

## Calibration and training

The paired Full-KV cache and 1024/128/128 manifests are reused. RMS scales are
recomputed from the train split because source and target directions are
reversed:

```text
source scales: Qwen3-4B cache (target4 family)
target scales: Qwen3-1.7B cache (source17 family)
```

Stage A trains from scratch using only KV reconstruction. Stage B initializes
strictly from the reverse Stage-A best checkpoint and uses only final-position,
full-vocabulary `KL(native1.7B || writer1.7B)`. The Receiver is frozen, gold
labels are excluded from training, effective batch is 8, gradient clip is 30,
maximum training is 4 true epochs, and early-stopping patience is 2 epochs.

## Evaluation

Native Qwen3-1.7B KV is the functional anchor. The formal evaluation includes
correct, true-zero, and exact-length shuffled controls, plus standard/full-text
references for both models. `comparison_to_forward_full28.json` reports
direction-normalized comparisons including accuracy retention, native
agreement, and native choice KL.

```text
artifacts/calibration/summary.json
checkpoints/quick/full36/stage_a/summary.json
checkpoints/quick/full36/stage_b_final/summary.json
artifacts/evaluation/formal/summary.json
artifacts/evaluation/formal/per_sample_generations.jsonl
artifacts/comparison_to_forward_full28.json
```

Run:

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```
