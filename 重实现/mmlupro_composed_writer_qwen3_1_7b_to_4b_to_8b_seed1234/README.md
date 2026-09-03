# 1.7B→4B→8B composed Writer audit

Zero-training composition audit over three existing 1.7B→4B Writers and
three existing Full36+Head8 4B→8B checkpoints. Native8, Native4→8, chained,
and true-zero controls are evaluated on the same MMLU-Pro test samples.

Translated intermediate 4B KV is streamed in memory and is never persisted.
Only scalar/per-layer metrics and per-sample predictions are saved.
