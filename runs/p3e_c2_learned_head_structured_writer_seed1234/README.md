# P3-E-C2 Learned Head-Structured Writer

P3-E-C2 replaces the fixed C0 duplicate writer with a trainable Qwen3-8B sender-specific `8 -> 16` overcomplete head expansion while preserving layer, token, and 128-dimensional head structure.

## Writer

- Input: evidence-only Qwen3-8B Native KV `[16,T,8,128]`.
- Output: Canonical head bus `[16,T,16,128]`.
- K and V share one per-layer `16x8` head route.
- Each Canonical head has separate rank-32 K and V residual coordinate adapters.
- Initialization is exactly equal to `duplicate_writer16`.
- There is no token mixing, layer mixing, dimension compression, or Receiver-shaped output head.

Writer training freezes Qwen3-4B, the P3-E-B Native teacher Reader, and the P3-E-C1 Canonical Reader. The objective is answer NLL, hard-shuffled dependency margin, Native token-route/readout distillation, and a small head-diversity regularizer.

## Fresh Reader test

After Writer training, the Writer is frozen and the P3-E-C1 Reader used during Writer training is discarded. A fresh Canonical Reader is initialized only from the P3-E-B Native Reader and trained independently. It never loads P3-E-C1 parameters.

The full run continues through all diagnostic stages even when an overfit gate is below threshold; gate files remain available for interpretation.

```bash
bash /home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234/run_all.sh all
```
