# OpenBookQA frozen Stage-A Base + Residual64

Controlled comparison against shared-parameter Stage B under the frozen standard protocol.

- Reuses the exact Full28-MLP Stage-A best-validation-accuracy checkpoint (step312).
- Freezes and hashes the Base before/after Stage B.
- Trains only independent K/V, layer/head-specific `128 -> 64 -> 128` GELU residual adapters.
- The up projection is zero-initialized, so the initial output is exactly the Stage-A Base output.
- Uses official OpenBookQA Main splits with deterministic seed1234 subsets: 1024/128/128.
- Fair A/B: both branches start from the same Stage-A checkpoint and receive 512 Stage-B optimizer steps (4 true epochs, batch 8, A-D choice-KL).
- Shared branch updates the complete Full28-MLP; decoupled branch freezes it and trains only Residual64.
- Reports Base/shared/decoupled metrics, paired correctness, and K/V residual norm diagnostics.
