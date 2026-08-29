# Full-KV Native Reader 1.7B → 4B: Results

Experiment: `fullkv_native_reader_qwen3_1_7b_to_4b_seed1234`

## Run status

- Seed: 1234
- Train / validation / test: 64 / 16 / 32 HotpotQA examples
- Input protocol: full context KV (not sparse supporting-token KV)
- Receiver: frozen native Qwen3-4B reader
- Writers: D0, D1 and D2, with independent per-target-layer K/V mappings
- Stage A / Stage B budget: 150 / 150 steps per writer
- All 16 ordered pipeline stages completed on 2026-08-29.

## Protocol audit

The protocol audit passed all checks on 8 examples. The official cache path and the manual post-RoPE cache extraction path matched exactly. Full-text and split-cache forward paths passed the explicitly declared FP16 split-forward tolerances, including logits, answer NLL and greedy generation checks.

## Final evaluation (32 examples)

| Condition | EM | F1 | Answer NLL |
|---|---:|---:|---:|
| 1.7B full text | 0.0000 | 0.0540 | 14.5063 |
| 4B full text | 0.0000 | 0.1256 | 10.5471 |
| 4B native full KV | 0.0000 | 0.1256 | 10.5477 |
| Question only | 0.0000 | 0.0747 | 12.1390 |
| Writer zero | 0.0000 | 0.0612 | 14.3288 |
| D0 correct | 0.0000 | 0.0115 | 12.0332 |
| D1 correct | 0.0000 | 0.0237 | 11.3941 |
| D2 correct | 0.0000 | 0.0247 | 9.7326 |

Permutation controls:

| Writer | Correct F1 | Token-permuted F1 | Correct − permuted F1 | Correct − zero F1 |
|---|---:|---:|---:|---:|
| D0 | 0.0115 | 0.0233 | -0.0118 | -0.0497 |
| D1 | 0.0237 | 0.0436 | -0.0199 | -0.0375 |
| D2 | 0.0247 | 0.0169 | +0.0079 | -0.0365 |

## Interpretation

The cache protocol itself is validated: native 4B full KV reproduces the 4B full-text baseline within the declared numerical tolerance. Under this deliberately small validation budget, however, none of D0/D1/D2 recovers the native 4B functional behavior. D2 gives the best writer NLL and a small positive F1 gap over its token-permuted control, but its F1 remains below the true-zero, question-only and native-KV baselines. Therefore this run validates the implementation and evaluation chain, but does not establish successful 1.7B → 4B Full-KV translation.

The complete machine-readable summaries, per-example generations, diagnostics, training histories and logs are included under `artifacts/` and `logs/`. KV caches, model checkpoints, calibration tensors and dataset copies are intentionally excluded.
