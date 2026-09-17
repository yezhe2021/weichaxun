# Qwen3-4B to Gemma3-4B reverse KV onboarding

This is the protocol-matched reverse of the Gemma3-4B to Qwen3-4B Full34 HeadMix256 experiment.

- Direction: Qwen3-4B Sender -> Gemma3-4B Receiver.
- Dataset: official OpenBookQA Main, deterministic 4096/128/128 subsets, seed 1234.
- Prompt: `Question -> Options -> Answer:`; receiver retains Gemma-native Question KV.
- Selection: RawAnchor32 over Options only using the final Qwen `Answer:` query.
- Source cache: pre-RoPE Qwen KV `[36,32,8,128]`.
- Writer: per-target-layer and K/V-independent depth MLP `4608 -> 1024 -> 128`, followed by full learned head mixing `8*128 -> 4*256`, all bias-free and with no token mixing.
- Gemma Receiver uses float32 on V100 and applies its exact sliding/full layer-specific RoPE only after translated pre-RoPE KV injection.
- Stage A: 2048 updates of KV reconstruction; validation functional accuracy selects the checkpoint.
- Stage B: both frozen-base Residual64 and fully shared fine-tuning, 1024 updates each with choice-only KL.
- Diagnostics: native baselines and controls, learned `[34,4,8]` head correspondence and `[34,36]` layer correspondence.
