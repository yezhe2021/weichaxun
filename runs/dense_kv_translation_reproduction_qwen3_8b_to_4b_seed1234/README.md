# Dense KV translation reproduction: Qwen3-8B → Qwen3-4B

This directory is an independent reproduction experiment. It uses only the
specified local Qwen3 models and GSM8K data. It does not import Canonical,
Writer, Reader, compression, Top-K, trajectory correction, or any earlier
experiment module.

The pipeline is deliberately split so the 8B sender, 4B receiver, and
translator never have to reside on the V100 together:

1. deterministic GSM8K split;
2. native-cache injection oracle and RoPE round-trip (Gate 0);
3. offline sender KV extraction, then offline receiver KV extraction;
4. translator-only reconstruction training (Phase I and Gate 1);
5. Phase-I generation sanity test;
6. offline receiver-self trace generation;
7. frozen 4B plus translator generation adaptation (Phase II);
8. fixed-test evaluation.

Every K and V translator is independent for every `(layer, KV group)` and has
the exact `128 → 2048 → 128` MLP shape. K and V parameters are not shared.
There is no token attention inside the translator. The scalar reliability gate
is implemented as `sigmoid(alpha[layer, group])`. This gate parameterization is
a reproduction implementation detail because the paper does not fully specify
its parameterization.

KV tensors are stored as FP16 shards by sample, layer, and token chunk. No token
or layer compression is applied. The 8B and 4B token IDs, position IDs, padding
mask, layer count, KV-head count, and head dimension are asserted.

The orchestrator writes a machine-readable status file and applies Gate 0 and
Gate 1 automatically. A failed gate stops subsequent training without requiring
manual inspection. Phase I uses only K/V MSE. Phase II uses only generation CE
and initializes from the best Phase-I validation checkpoint.

Run:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py --config config.json
```

All reproduction hyperparameters not disclosed by the paper are recorded in
`config.json`.
