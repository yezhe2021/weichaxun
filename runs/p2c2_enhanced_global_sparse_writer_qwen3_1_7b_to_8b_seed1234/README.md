# P2-C2: Enhanced Global Sparse Writer

P2-C2 keeps the Qwen3-8B backbone and P2-A2 Query-only Reader strictly frozen
and trains only a Qwen3-1.7B sender-specific Writer. It reuses the exact P2-C1
data and Native-KV caches.

## Architecture

- Independent global K/V routing from all sender layers to every receiver layer.
- Learned relative-depth bias with dense warmup and deterministic top-6 routing.
- Independent K/V head mixing.
- Per-receiver-head rank-32 residual adapters with zero-initialized up matrices.
- Per-layer/head scales and Native 8B K-RMS calibration.
- Complete evidence tokens and complete per-layer K/V; no compression.

## Training

`full_staged` runs three one-epoch stages:

1. K-first: route KL, target attention mass, shuffled route ranking, and a small task term.
2. V-second: frozen K, Native readout alignment, and base/CF NLL.
3. Joint: small learning rate, generation, answer-swap and negative-memory ranking,
   plus low-weight route/readout/KV auxiliaries.

Every stage creates an optimizer containing only the active Writer parameters.
Receiver and Reader gradients are asserted absent after every backward, and the
inactive K/V Writer half is compared exactly at each stage boundary.

## Ablations

- `global_only`: global routing with shared full projections.
- `global_head`: global routing plus per-head adapters, joint training.
- `full_staged`: global routing, per-head adapters, K/V/joint curriculum.

The default `all` target runs only `full_staged`. Use `all-ablations` to train
all three configurations.

## Scope

The output remains Qwen3-8B Reader-specific per-layer KV and is not a
receiver-independent Canonical Evidence-KV.

```bash
cd /home/yezhe/伪查询
bash runs/p2c2_enhanced_global_sparse_writer_qwen3_1_7b_to_8b_seed1234/run_all.sh all
```
