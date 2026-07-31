# Native Qwen3-4B full-cache equivalence audit

No Sender and no training are present in this experiment. It verifies the
target 4B KV path before any additional cross-model work.

Eight conditions share the exact R1 prompt and answer protocol:

- full text, LoRA off;
- official base-model full cache, LoRA off/on at read time;
- manually reconstructed full cache, LoRA off/on at read time;
- sparse gold native KV, LoRA off/on;
- full text with LoRA enabled throughout.

The audit records prompt splitting, post-RoPE official/manual cache equality,
first-question-token layer trajectories, answer-token distribution comparisons,
and generation metrics. No hard pass/fail threshold is applied.
