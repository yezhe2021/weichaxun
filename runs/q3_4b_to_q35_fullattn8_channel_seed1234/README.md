# Qwen3-4B to Qwen3.5 Full-Attention Anchor8 channel

This experiment reverses the previous heterogeneous translation direction.
Qwen3-4B contributes only layers `[3,8,12,17,21,26,30,35]`; eight independent
bias-free full-rank K/V maps write into Qwen3.5 Full-Attention layers
`[3,7,11,15,19,23,27,31]`.

The Qwen3.5 external Anchor memory is static and separate from its official
HybridCache. The 24 DeltaNet layers never receive fabricated Context state and
continue to accumulate only Question/generated-token state.

Pipeline:

1. P0 full text / official HybridCache / native Anchor8 audit.
2. Conditional sender-independent Qwen3.5 Anchor8 Reader registration.
3. Character-offset Qwen3-to-Qwen3.5 supporting-token alignment and RMS scales.
4. Stage A 16-sample overfit and 512-sample representation alignment.
5. F1 identity-initialized and F2 Stage-A-initialized answer-only training.
6. Best-NLL and best-generation checkpoints plus unified evaluation.

Stage B has no representation tether or auxiliary loss.
