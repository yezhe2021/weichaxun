# P3-D Frozen Multi-Layer Canonical Writer to Qwen3-4B Real Reader Onboarding

This experiment freezes the Qwen3-8B sender, the selected P3-C writer, and the complete Qwen3-4B backbone. Only a layer-wise external Reader is trained.

## Protocol

- The main memory is the stable P3-C `uniform16` question-independent Canonical Writer output, `[16, T_E, 256]` for both K and V.
- Stable means the seed whose validation retention is closest to the median across the three P3-C seeds. The selected writer SHA-256, canonical layer order, and cache hashes are audited before training.
- `native16` uses the corresponding 16 Qwen3-8B native layers with all eight KV heads flattened to 1024 dimensions. It is the Reader capacity upper bound.
- `canonical36` is a wider P3-C capacity control.
- Every one of the 36 frozen Qwen3-4B decoder layers has an independent Reader. It attends within each external layer channel, routes across channels using the current hidden state and fixed layer embeddings, projects the readout to the residual dimension, and injects it through a learned near-zero gate.
- The same hook-based Reader path is used by teacher-forcing and `generate()`. External memory is immutable and is never added to the receiver `past_key_values`.

## Training

The deterministic ten-example schedule is 50% correct memory, 20% answer-length-matched shuffled memory, 10% zero memory, 10% K/V mismatch, and 10% Reader-off. Incorrect memories explicitly target `INSUFFICIENT`. Correct examples additionally receive answer-token logit KL from the frozen Qwen3-4B full-text path. Optimizer parameter identity and frozen-backbone gradients are checked.

A 16-example overfit run and free-running gate execute first. Per the project-wide instruction to finish diagnostic pipelines even when an intermediate metric misses its target, the gate records failure without terminating the full Native/Canonical comparison.

## Evaluation

Each trained Reader is evaluated under correct, shuffled, zero, K-correct/V-wrong, K-wrong/V-correct, K-correct/V-zero, K-zero/V-correct, canonical-layer permutation, synchronized token permutation, Reader-off, and question-only conditions. Outputs include raw generations, HotpotQA-style EM/F1, bridge/comparison slices, rejection accuracy, source-memory answer hits, EOS, gate/router diagnostics, memory size, Reader parameter count, and elapsed generation time.

Text baselines include Qwen3-4B question-only, Qwen3-4B gold full text, and Qwen3-8B gold full text. Existing P3-A old synthetic-Reader generations are filtered to the exact P3-B/P3-C extractive test IDs when available.

## Commands

```bash
bash run_all.sh all
bash run_all.sh status
tail -f p3d_run.log
```
