# P2-A: Full Native-KV External Reader Upper Bound

This directory contains two deliberately separate experiments.

## P2-A0: scalar-gate path check

The original `smoke` target trains only one scalar gate per layer. It checks that
the external branch executes, but it is not the complete content-reading test.

## P2-A1: complete native-KV Reader test

P2-A1 keeps the Qwen3-8B sender and receiver frozen and uses all 36 layers of
uncompressed, same-model, evidence-only pre-RoPE K and native V. The trainable
path consists of a rank-32 low-rank correction per layer and a small scalar gate.

The receiver prompt is prefixed with `FINAL:` so training and free-running start
at the answer position. Training uses strict B/B' pairs and optimizes both target
NLL and a counterfactual margin. The diagnosis reports:

- held-out NLL for y and y' under both B and B';
- positive-margin and paired-margin success rates;
- per-layer B/B' readout and injected-delta distances;
- attention mass assigned to the answer-bearing evidence tokens;
- free-running correct, counterfactual, shuffled, mismatched, zero, and Reader-off results.

No heterogeneous mapping, canonical space, compression, or multi-sender logic is
included in P2-A1.

## Default quick configuration

```text
train: 64 counterfactual pairs
test: 16 counterfactual pairs
epochs: 2
Reader rank: 32
generation: 24 tokens
```

Run the complete P2-A1 pipeline:

```bash
cd /home/yezhe/伪查询
bash runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/run_all.sh a1
```

The A1 cache is separate because it includes an answer-token mask required by
the content diagnosis. Do not reuse the old A0 cache.

Inspect the main outputs:

```bash
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/text_gate/SUCCESS.json
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/train_a1_rank32/TRAIN_SUCCESS.json
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/diagnose_a1_rank32/SUCCESS.json
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/eval_a1_rank32/SUCCESS.json
```
