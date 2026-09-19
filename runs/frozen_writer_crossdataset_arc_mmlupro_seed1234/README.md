# Frozen Writer cross-dataset transfer

Zero-shot evaluation of OpenBookQA-trained frozen KV Writers on ARC-Challenge and
MMLU-Pro.  No parameter is trained or selected on either target dataset.

The suite evaluates both directions and both path lengths:

- Llama3.2-3B -> Qwen3-4B and Qwen3-4B -> Gemma3-4B;
- Qwen3-4B -> Llama3.2-3B and Gemma3-4B -> Qwen3-4B;
- Llama -> Qwen Hub -> Gemma;
- Gemma -> Qwen Hub -> Llama.

All routes use the unchanged standard `Question -> Options -> Answer:` protocol,
32 normalized option-text regions, receiver-native Question KV, and frozen checkpoints.
