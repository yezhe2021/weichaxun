# Reverse Qwen3-4B to Llama3.2-1B onboarding

- Direction: Qwen3-4B Sender -> Llama3.2-1B Receiver.
- Official OpenBookQA Main 4096/128/128, standard prompt, Options-only RawAnchor32.
- Receiver-native Llama1B Question cache.
- Full36-MLP geometry: `[36,T,8,128] -> [16,T,8,64]` using `4608 -> 1024 -> 64` per target layer/head.
- K/V are separate; target layers are independent; heads and tokens never mix; all linear layers are bias-free.
- Stage A: K/V reconstruction, 4 epochs / 2048 steps.
- Stage B: frozen Stage-A plus Llama1B-space Residual64, 2 epochs / 1024 steps of choice KL.
- Shared Stage-B is omitted.
- Baselines include Qwen Full Native/Selected32, Llama1B Full Native/Oracle32, zero, no-memory, and shuffled-memory controls.
- Final output reports the Llama1B Native Oracle32 minus translated accuracy gap.

The complete pipeline is run with:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study all
```
