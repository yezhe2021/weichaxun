# P2-I-W Token-Preserving Canonical Writer Bootstrap

This experiment isolates the Qwen3-8B sender Writer from every receiver. It tests whether final-layer,
evidence-token Native K/V can be transformed into a sample-specific, independently readable canonical
memory without global slot pooling or token compression.

## Fixed protocol

- Reuses the exact P2-A2 512-train-pair and 64-test-pair Native-KV caches.
- Uses pair-level `448/64/64` train/validation/test splits.
- Extracts only final-layer evidence-token pre-RoPE K, native V, and post-final-norm hidden teacher states.
- Flattens 8 KV heads into 1024 dimensions and preserves the variable evidence-token axis.
- Emits K/V `[T_E, 256]`; it has no layer, head, RoPE, slot, or receiver axis.
- Fits PCA/random projections on the 448-pair training prefix only.
- Uses a hidden-teacher probe gate and a 16-pair Writer overfit gate before full training.
- Discards the training probe and trains a fresh attention probe against a frozen Writer.

## Stages

```bash
bash run_all.sh audit
bash run_all.sh cache
bash run_all.sh projections
bash run_all.sh baselines
bash run_all.sh small
bash run_all.sh train
bash run_all.sh fresh-probe
bash run_all.sh diagnostics
bash run_all.sh summarize
```

Run the gated pipeline with:

```bash
bash run_all.sh all
```

The default pipeline records a warning and continues if the final-hidden teacher probe or the 16-pair Writer
overfit positive control misses its threshold. This permits end-to-end diagnosis without treating a failed
positive control as success. Set `STRICT_GATES=1` to restore hard-stop behavior.

## Controls and outputs

The fresh probe reports correct, base/CF memory swap, cross-sample shuffled memory, current-K/other-V,
zero memory, and synchronous token permutation. Swap controls report both original-target and source-memory
accuracy. Diagnostics report K/V/shared effective rank, K/V CKA and relation-graph agreement, corresponding
token variance/cosine, pooled cross-sample cosine, and changed versus unchanged base/CF span distances.

Checkpoints and tensor caches are intentionally not suitable for Git. Commit scripts and JSON/JSONL results;
exclude `.pt`, cache shards, and logs.
