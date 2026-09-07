# Native token0 + trained 16-slot audit

This no-training audit reuses the best checkpoints from the 16-slot soft-locality study and
tests whether adding the original Native context token0 restores self-slot function.

The clean comparison controls the position shift:

- `native_full`: complete Native context KV.
- `slot16_original`: 16 slots at positions 0..15; receiver suffix begins at 16.
- `slot16_shifted_control`: the same slots at positions 1..16; suffix begins at 17.
- `token0_plus_slot16`: Native token0 at position 0 plus the same slots at 1..16; suffix begins at 17.

Therefore the difference between `token0_plus_slot16` and `slot16_shifted_control` isolates
the immediate effect of adding token0. Models and slot modules are frozen. Qwen3-4B and
Llama-3.2-3B use the same 128 MMLU-Pro test samples as the preceding studies.

```bash
cd /home/yezhe/异构模型/selfslot16_plus_native_token0_audit_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/formal/status.json
```
