# Qwen3-8B→4B Full-36 + Head-8 Writer

Direct main experiment with no separately trained layer-only baseline. Full-context pre-RoPE K and V are used.
All 36 source layers are concatenated per KV head, then all eight depth-projected heads are concatenated.
Every target layer/head owns independent bias-free K/V head projections. Both depth and head stages train.

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```
