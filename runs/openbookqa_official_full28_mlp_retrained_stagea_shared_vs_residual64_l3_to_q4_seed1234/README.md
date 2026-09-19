# Official OpenBookQA: Fresh Stage-A + Shared vs Residual64

Leakage-corrected experiment using only the official OpenBookQA Main split boundaries.

- Train/validation/test: deterministic 1024/128/128 subsets of official train/dev/test.
- Reuses only protocol-identical KV and Oracle-teacher caches from the preceding official-data run.
- Does **not** reuse any Stage-A checkpoint or optimizer state.
- Stage-A: randomly initialized Full28-MLP, KV reconstruction, 8 true epochs / 1024 steps.
- Stage-A checkpoint selection: official validation128 only.
- Shared Stage-B: update all Full28-MLP parameters for 4 epochs / 512 steps.
- Decoupled Stage-B: freeze the newly trained Stage-A and train independent K/V, layer/head-specific Residual64 adapters for 4 epochs / 512 steps.
- Both Stage-B branches use the same final-position A-D choice-KL objective.
- Test evaluation reports accuracy, Oracle agreement/KL, K/V similarity, residual norms, and paired correctness counts.

No AraDiCE-derived checkpoint is loaded anywhere in this pipeline.
