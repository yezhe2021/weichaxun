# Llama3.2-3B → Qwen3-4B Hub → Gemma3-4B

Evaluation-only composability audit on the same 128 official OpenBookQA test examples used by
the two frozen train4096 writers. It compares Llama→Qwen Stage-A and residual Stage-B Hub
states after both are consumed by the same Qwen→Gemma writer.

Controls include a Qwen-native bridge and Gemma-native Oracle32 at the same Llama-selected raw
anchors. Both downstream Stage-A and residual branches are evaluated. No parameter is trained.
