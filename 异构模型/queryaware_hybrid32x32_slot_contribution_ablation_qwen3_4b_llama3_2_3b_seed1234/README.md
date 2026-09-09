# Query-aware Hybrid32x32 Slot Contribution Ablation

Evaluation-only causal ablation using the frozen Query selector and best Query-arm slot checkpoint from `queryaware_hybrid32x32` on the same 128 MMLU-Pro test rows.

- `compact_native_only`: `[token0,N0,...,N31]`, positions 0..32, suffix starts at 33.
- `position_matched_native_only`: `[token0,N0,masked0,...,N31,masked0]`, positions 0..64, suffix starts at 65.
- `real_slot_hybrid`: `[token0,N0,S0,...,N31,S31]`, positions 0..64, suffix starts at 65.

The primary contrast is Real Hybrid minus Position-matched Native-only. No parameters are trained.
