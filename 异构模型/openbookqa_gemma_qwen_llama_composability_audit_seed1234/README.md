# Gemma → Qwen → Llama composability audit

This is an evaluation-only experiment. It freezes the completed Gemma3-4B → Qwen3-4B
Stage-A and residual Stage-B writers, then passes each Qwen-space output through the same
completed Qwen3-4B → Llama3.2-3B reverse writer.

The primary comparison is `gemma_stage_a` versus `gemma_stage_b_residual`. A Qwen-native
bridge and a Llama-native Oracle32 at the same raw-text anchors separate upstream Hub drift
from limitations of the reverse writer and receiver layout.

Both reverse Stage-A and reverse residual branches are reported. No module is trained.
