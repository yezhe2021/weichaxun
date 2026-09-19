# Qwen3.5 Anchor-only Sender

This experiment reuses the validated v2 sample IDs, character-offset alignment,
Qwen3.5 caches, source/target scales, A1 checkpoint, and R1 sparse Reader.

Only eight bias-free full-rank K/V feature maps are retained. External memory is
provided at Qwen3-4B layers `[3,8,12,17,21,26,30,35]`; the other 28 layers have
no external slot. External memory is held separately from the Receiver's normal
DynamicCache, so Question and generated-token self-cache still accumulates at
all 36 layers.

Execution order:

1. Protocol/assets audit.
2. Native4 Full36/Anchor8 upper-bound evaluation with the frozen current Reader.
3. Conditional Native4-only Anchor8 Reader adaptation when the upper bound collapses.
4. Anchor-F1 identity initialization and answer-CE training.
5. Anchor-F2 A1 initialization and answer-CE training.
6. Unified free-running evaluation and imported Full36 references.

No hard result gate stops the pipeline.
