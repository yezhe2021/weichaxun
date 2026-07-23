# P3-E-C1 Learnable Overcomplete Canonical Head Reader

This stage keeps the P3-E-C0 `duplicate_writer16` fixed and lossless. It trains only a receiver-specific Qwen3-4B Reader on the same train512/validation64 HotpotQA split used by P3-E-B.

## Path

1. Qwen3-8B evidence-only Native KV: `[16,T,8,128]`.
2. Frozen duplicate writer: `[16,T,16,128]`, with each Native head copied to two adjacent Canonical heads.
3. Qwen3-4B native pre-RoPE Query plus rank-32 residual Query adapter: `[B,S,32,128]`.
4. Independent token attention from every Query head to every Canonical head.
5. Per-layer trainable `32x16` route, initialized to the exact C0 duplicate mapping and executed with straight-through top-2 routing.
6. The 32 head readouts are flattened to 4096 dimensions, passed through the frozen native `o_proj`, multiplied by one scalar gate per selected layer, and added as an external attention branch.

No Writer parameters, Receiver backbone parameters, layer router, cross-layer fusion, compatibility gate, output MLP, or token/dimension compression are trained.

## Loss

`answer-token mean NLL + 0.5 * relu(0.5 + correct NLL - hard-shuffled NLL)`

The run first checks exact C0 initialization equivalence, then runs overfit16 and its automated gate, followed by formal512 and validation64 only if the gate passes.

## Run

```bash
bash /home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234/run_all.sh all
```
