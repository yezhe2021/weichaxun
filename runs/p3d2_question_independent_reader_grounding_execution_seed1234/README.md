# P3-D2 Question-Independent Reader Grounding and Execution Compatibility

This run preserves the frozen Qwen3-8B sender, the selected P3-C `uniform16` Writer, the extractive HotpotQA split, and the frozen Qwen3-4B backbone.

## Executed scope

1. **Oracle grounding diagnosis:** reuse the P3-D Reader checkpoint and compare ordinary reading, gold answer-token masking, and gold token plus top-four P3-C span-probe layer masking. The model and generation path are unchanged.
2. **Grounded execution training:** cache relative answer-position traces from frozen Qwen3-4B question-only and full-text paths. Train only the external Reader using answer NLL, frozen span-probe grounding KL, receiver-native residual/logit-delta alignment, hard question-memory compatibility ranking, and natural residual-scale matching. Shuffled memory is not trained to generate `INSUFFICIENT`; zero memory is only a question-only control.
3. **Capacity comparison:** train `uniform8`, `midlate8`, `key4`, and `all36`. Projections and routers are shared in groups; every active receiver layer retains an independent gate and all variants use an explicit compatibility gate. Checkpoints are selected by validation free-running correct F1, bridge F1, correct-shuffled gap, and compatibility accuracy.

Step 4 (Receiver LayerNorm or rank-8 LoRA adaptation) is intentionally not executed and is absent from the run entry point.

```bash
bash run_steps_1_3.sh all
bash run_steps_1_3.sh status
tail -f p3d2_run.log
```
