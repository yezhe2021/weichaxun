# MMLU-Pro Full-28 Writer Functional Upper Baseline

This experiment changes only the Writer layer receptive field relative to the
locked Local-5 B1 protocol. Data, Full-KV cache, normalization, Receiver,
objectives, sampler, optimizer settings, and evaluation are unchanged.

## Writer

For each K/V family, source layer is concatenated independently at every token
and KV head:

```text
[28,T,8,128] -> [T,8,28*128] -> 36 independent Linear(3584,128)
```

The invariants are enforced by unit tests and checkpoint metadata:

- each target layer owns an independent linear map;
- K and V parameters are separate;
- every target layer uses the same ordered source layers 0..27;
- no token mixing and no head mixing;
- every linear layer has `bias=False`;
- `Writer(0)=0`;
- initialization exactly reproduces the calibrated nearest-layer D0 mapping.

## Training

Stage A is trained from scratch with only the existing KV reconstruction loss.
Its best validation loss is the representation comparison against Local-5.

Stage B initializes strictly from this experiment's Full-28 Stage-A best
checkpoint and uses only final-position, full-vocabulary
`KL(native || writer)`. It retains the true epoch sampler, 1,024 train samples,
effective batch 8, at most 4 epochs, patience 2, and gradient clip 30.

## Outputs

```text
checkpoints/quick/full28/stage_a/summary.json
checkpoints/quick/full28/stage_a/best_layer_metrics.json
checkpoints/quick/full28/stage_b_final/history.json
checkpoints/quick/full28/stage_b_final/epoch_summaries.json
checkpoints/quick/full28/stage_b_final/summary.json
artifacts/evaluation/formal/summary.json
artifacts/evaluation/formal/per_sample_generations.jsonl
artifacts/comparison_to_local5.json
```

Run:

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```
