# P2-I-R Token-Preserving Canonical Evidence-KV Reader Onboarding

This experiment freezes the validated P2-I-W Qwen3-8B Writer permanently and tests whether real frozen
Qwen Receivers can use the same variable-length `[T_E,256]` Canonical Evidence-KV for free-running answers.

## Invariants

- The P2-I-W Writer is loaded once to cache train/test Canonical memory and never enters Reader training.
- Cache manifests record both the Writer checkpoint file SHA-256 and the Writer state SHA-256.
- Qwen3-4B and Qwen3.5-4B consume exactly the same cache format and Writer hashes.
- Receiver backbones, embeddings, native attention, and LM heads are frozen; optimizer ID audits allow only Reader parameters.
- Qwen3-4B reads at all 36 layers. Qwen3.5-4B reads only at its eight standard full-attention layers.
- No Native-KV MSE, Writer update, q-aware selection, layer/head/token compression, or multi-Sender composition is used.

Each Receiver runs a 16-pair overfit training/evaluation followed by 448/64 full training/validation and a
64-pair held-out free-running evaluation. Missing thresholds are recorded but never stop the next stage.

Controls include correct memory, base/CF whole-memory swap, cross-sample shuffled memory, current-K/other-V,
zero memory, Reader-off, and synchronous K/V token permutation. Every generation and layer diagnostic is saved.

```bash
bash run_all.sh all
bash run_all.sh status
```

Tensor caches, model checkpoints, and logs should not be committed to Git. Commit scripts and JSON/JSONL results.
