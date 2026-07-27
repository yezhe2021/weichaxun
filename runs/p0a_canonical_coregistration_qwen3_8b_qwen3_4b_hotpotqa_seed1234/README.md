# P0-A: Canonical Co-registration

This is a standalone Qwen3-8B/Qwen3-4B experiment. It tests only whether two
senders can write their native context KV into a functionally interchangeable
canonical space. There is no Receiver, answer generation, selector, Top-K,
slot compression, token compression, layer compression, or head compression.

The fixed input order is `Context → Question`. HotpotQA gold supporting facts
provide the evidence positions. Eight fixed layers cover model depth:
`[0, 5, 10, 15, 20, 25, 30, 35]`. All 8 KV heads and all selected context
tokens remain present.

Each sender has independent K Writer, V Writer, and Query Adapter. Each adapter
uses a shared sender-local `128 → 256 → 128` residual MLP plus learned
layer/head embeddings and RMS normalization. K, V, and Q parameters are
separate. Qwen3 has 32 query heads and 8 KV heads, so the four native query
heads associated with each KV head are averaged before the Query Adapter. This
is an explicit reproduction implementation detail.

The shared objective is:

`L = L_retrieval + 0.1 L_Q + 0.1 L_K + 0.5 L_V`

where retrieval trains all AA/AB/BA/BB combinations, with cross combinations
weighted twice as strongly as self combinations. Retrieval negatives include
all non-supporting sentences and tokens in the same HotpotQA context. No KV
MSE, answer CE, Receiver KL, trajectory, reconstruction, or sender-ID
adversarial loss is used.

The pipeline first runs an 8-train/4-validation smoke test. Only after all
mechanical checks and both private/shared smoke training paths finish does it
continue to 256 balanced train and 64 balanced validation examples.

Large native KV caches and checkpoints are intentionally excluded from Git.

Run:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py --config config.json
```
