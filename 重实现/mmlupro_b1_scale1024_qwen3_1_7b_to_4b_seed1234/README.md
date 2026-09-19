# MMLU-Pro D2 B1 Scale-1024 Full-KV Experiment

This experiment tests whether the weak generalization of final-position behavioral distillation was caused by insufficient data and optimization. It intentionally removes B2 and locks the method:

```text
Writer:  D2 Local-5 Linear
Stage A: ONLY full K/V reconstruction
Stage B: ONLY final-position full-vocabulary KL(Teacher || Student)
```

The Sender sees original-order Options only. The frozen 4B Receiver receives the Question/instruction/`Answer:` suffix later. Stage B physically removes all gold fields, uses Native4 cache behavior as teacher, and never uses MMLU labels for training or checkpoint selection.

## Controlled scale-up

```text
train      1024
validation  128
test        128
```

- Test IDs and prefix hashes are exactly anchored to the preceding 128-sample pure-functional experiment.
- The preceding train-128 set is a strict subset of train-1024.
- The preceding validation-32 set is a strict subset of validation-128.
- Added samples follow the same deterministic category-balanced order and exclude all anchored test samples.
- Existing anchor caches are hard-linked; missing caches are generated normally.

To preserve approximately the same number of sample exposures as the preceding train-128 experiment:

```text
Stage A:  8,000 updates, validation every 400
Stage B1: 16,000 updates, validation every 400
```

Both use the same learning rates as before (`1e-4` and `3e-5`). Stage B1 initializes only from the new Stage-A best checkpoint.

## Diagnostics and evaluation

KV reconstruction and 36-layer hidden trajectory metrics remain diagnostics only during Stage B. Formal evaluation reports Native4, Question-only, Stage A, B1 Correct, true-zero, and exact-length matched Shuffled controls. Correct-vs-Shuffled deltas are computed only on the same eligible sample IDs.

## Run

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```

Shared/full cache tensors and `.pt` checkpoints remain server-local and are excluded from Git.
