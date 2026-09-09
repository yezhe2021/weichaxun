# MMLU-Pro B1 True-Epoch Sampler + Batch-8 Accumulation

This experiment changes only Stage-B optimization. It reuses the exact train-1024/validation-128/test-128 manifests, paired Full-KV cache, RMS calibration, and D2 Stage-A best checkpoint from `mmlupro_b1_scale1024_qwen3_1_7b_to_4b_seed1234`.

Locked method:

```text
Writer:  D2 Local-5 Linear
Stage A: reused KV-reconstruction best checkpoint
Stage B: ONLY final-position full-vocabulary KL(Teacher || Student)
```

## Stage-B sampler and optimization

```text
train samples                1024
micro batch size                1
gradient accumulation steps     8
effective batch size             8
optimizer steps per epoch      128
maximum epochs                   4
maximum optimizer steps        512
maximum sample exposures      4096
early-stopping patience          2 epochs
validation cadence               1 epoch
```

Every epoch uses a deterministic, newly shuffled permutation of all 1,024 indices. Each sample appears exactly once per completed epoch; sampling with replacement is forbidden and asserted at runtime. Each optimizer step averages eight independently forwarded sample losses before gradient clipping and `optimizer.step()`.

## Clip-rate logging

`torch.nn.utils.clip_grad_norm_` returns the total norm before clipping. Every optimizer-step record contains:

- the eight sample IDs;
- mean functional KL;
- pre-clip gradient norm;
- clipping threshold;
- whether clipping occurred.

Each epoch and the final summary report `clipped_optimizer_steps` and `clip_rate`.

## Early stopping

Validation final-position KL is measured after each epoch and is the only checkpoint-selection metric. If it fails to improve for two consecutive completed epochs, training stops. Accuracy and gold labels are never used for training, early stopping, or checkpoint selection.

## Outputs

```text
checkpoints/quick/d2/stage_b_final/history.json
checkpoints/quick/d2/stage_b_final/epoch_summaries.json
checkpoints/quick/d2/stage_b_final/summary.json
artifacts/evaluation/formal/summary.json
artifacts/evaluation/formal/per_sample_generations.jsonl
```

Run:

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```
