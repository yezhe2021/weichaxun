# Official OpenBookQA forward Full28-MLP train4096

This experiment changes only the amount of clean forward training relative to the train1024 baseline.

- Sender: Llama3.2-3B; Receiver: Qwen3-4B.
- Official OpenBookQA Main train/dev/test splits with deterministic 4096/128/128 selection.
- Standard `Question -> Options -> Answer:` prompt.
- Options-only region-balanced 32-token routing and RawAnchor alignment.
- Receiver-native Question cache.
- Full28-MLP with independent target layer/head mappings and separate bias-free K/V parameters.
- Stage A: fresh initialization, KV reconstruction, 4 epochs / 2048 steps.
- Shared Stage B: Choice-KL, 2 epochs / 1024 steps.
- Residual64 Stage B: frozen Stage-A base, Choice-KL, 2 epochs / 1024 steps.
- Validation every 256 steps; final comparison on the unchanged test128 split.

No previous checkpoint, optimizer state, or training cache is reused.
