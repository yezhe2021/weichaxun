# Gemma3-4B onboarding into the fixed Qwen3-4B KV Hub

This independent experiment changes only the Sender and the required head geometry relative to the established OpenBookQA onboarding protocol.

- Direction: Gemma3-4B Sender -> Qwen3-4B Receiver.
- Dataset: official OpenBookQA Main, deterministic 4096/128/128 subsets, seed 1234.
- Prompt: `Question -> Options -> Answer:`; receiver retains Qwen-native Question KV.
- Selection: RawAnchor32 over Options only using the final Sender `Answer:` query.
- Source cache: pre-RoPE Gemma KV `[34,32,4,256]`, with exact sliding/full layer-specific RoPE used only for routing and native readout.
- Writer: per-target-layer and K/V-independent depth MLP `8704 -> 1024 -> 256`, followed by full learned head mixing `4*256 -> 8*128`, all bias-free and with no token mixing.
- Stage A: 2048 updates of KV reconstruction; validation functional accuracy selects the checkpoint.
- Stage B: both frozen-base Residual64 and fully shared fine-tuning, 1024 updates each with choice-only KL.
- Diagnostics: native baselines and controls, learned `[36,8,4]` head correspondence and `[36,34]` layer correspondence.

Run:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u tests.py
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode smoke all
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study all
```
