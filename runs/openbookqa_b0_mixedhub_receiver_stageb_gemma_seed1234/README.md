# B0: Mixed-Hub receiver-specific Gemma Stage-B

This experiment freezes Llama3.2-3B → Qwen3-4B Stage-A, Qwen3-4B → Gemma3-4B Stage-A,
and all language models. It trains only one source-agnostic Gemma Residual64 using an exact
50/50 mix of Qwen-native and Llama3B-StageA Hub states.

For every training question, both Hub states use the same Llama-selected RawAnchor32 positions
and the same Gemma Native Oracle32 choice-logit teacher. No source ID, gate, source-specific
parameter, new loss, or changed selection policy is introduced.
