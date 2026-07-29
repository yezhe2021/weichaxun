# Full-Depth Anchor Canonical Sender Registration

Canonical v1 is fixed to the complete 36-layer Qwen3-4B sparse Native KV
protocol. Qwen3-4B's Writer is identity, and the already trained R0
`LoRA_self` Reader is frozen. The only trained component is the Qwen3-8B
Writer.

The Writer maps layers one-to-one with independent K/V identity-initialized
8x8 head mixers and per-layer `128 -> 256 -> 128` residual MLPs. It never
mixes layers or tokens and performs no compression.

Assets reuse the exact R0/R1 1024/128/128 sample IDs and full-context selected
pre-RoPE K/native V. This experiment adds offline 4B native Question queries
for route and attention-output warm-up.

```bash
cd /home/yezhe/可拔插/full_depth_anchor_canonical_sender_registration_qwen3_8b_to_4b_seed1234
tail -f logs/pipeline.log
```
