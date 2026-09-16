# Llama3.2-1B onboarding into the fixed Qwen3-4B KV Hub

This is a single end-to-end onboarding experiment, not a new method search.

- Direction: Llama3.2-1B Sender -> Qwen3-4B Receiver.
- Fixed standard `Question -> Options -> Answer:` prompt and official OpenBookQA Main 4096/128/128 split.
- Fixed Options-only RawAnchor32 routing and Receiver-native Question protocol.
- New source geometry only: `[16,T,8,64] -> [36,T,8,128]`.
- Full16-MLP: per target layer/head, separate bias-free K/V mappings, no head or token mixing.
- Stage A: 4 epochs / 2048 steps of K/V reconstruction.
- Stage B: frozen Stage-A plus Residual64, 2 epochs / 1024 steps of final A-D choice KL.
- Shared Stage-B is intentionally omitted and should only be added if Residual64 fails.
- Before training, the same pipeline evaluates Qwen Full Native, Llama1B Full Native, Llama1B Selected32,
  Qwen Native Oracle32, exact-zero memory, no memory, and cross-sample shuffled memory.
- Final output reports `Oracle32 accuracy - translated accuracy` as the onboarding gap.

Run the complete experiment with one command:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study all
```
