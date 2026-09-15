# Official OpenBookQA Frozen Base + Residual128

Rank-only follow-up to the leakage-corrected Residual64 experiment.

- Reuses the clean, freshly trained Stage-A step768 checkpoint from `openbookqa_official_full28_mlp_retrained_stagea_shared_vs_residual64_l3_to_q4_seed1234`.
- Does not train Stage-A.
- Does not run Shared Stage-B.
- Trains only independent K/V, layer/head-specific `128 -> 128 -> 128` receiver-space residual adapters.
- Zero-initialized output projections make the initial model exactly equal to the frozen Stage-A base.
- Official OpenBookQA Main: train1024/validation128/test128.
- Stage-B: 4 true epochs, batch8, 512 optimizer steps, final-position A-D choice-KL.
- Final result includes a paired Residual64-vs-Residual128 comparison on the identical test128 IDs.
