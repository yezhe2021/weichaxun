# Native 4B Channel Writer8

Evidence-only Qwen3-8B Native KV is translated directly into the frozen
Qwen3-4B Native Sparse KV channel. The question is only consumed by the frozen
4B Receiver.

Implemented variants:

- joint K/V linear baseline;
- per-layer/token/head residual MLP;
- residual MLP plus one head-interaction block.

Each variant runs Stage A representation alignment and Stage B frozen-Reader
functional calibration. R1's `sparse_reader/best.pt` is frozen and reused
without redesign.

Depth interaction, token interaction, question conditioning, layer/head
compression, new Receivers, and multi-Sender composition are intentionally
excluded.

```bash
cd /home/yezhe/可拔插/native4_channel_writer8_qwen3_4b_8b_hotpotqa_seed1234
tail -f logs/pipeline.log
```
