# Memory-first Receiver Oracle Audit

This experiment changes only Receiver readout order while fixing the standard Sender prompt,
Answer-query + 32-region routing, RawAnchor pairs, and Full28-Diagonal Translator.

The controlled comparison is:

```text
Question-first: token0 -> native Question -> 32 native option KVs -> native Answer:
Memory-first:   token0 -> 32 native option KVs -> native Question -> native Answer:
```

The first formal stage evaluates both native-Qwen Oracle layouts on the same 128 test examples.
The Stage-B continuation gate is fixed before evaluation: memory-first must improve by at least
3 correct examples (2.34 points). If it passes, the pipeline reuses the existing Stage-A
Full28-Diagonal checkpoint and trains exactly 512 memory-first choice-KL Stage-B steps. If it
does not pass, training is intentionally skipped. Smoke always exercises the training code for
one step, but never controls the formal gate.

No Sender/Router cache is regenerated. Existing manifests, selected Llama KVs, RawAnchor-aligned
Qwen KVs, and Stage-A checkpoints are read-only inputs.

```bash
cd /home/yezhe/异构模型/mmlupro_memoryfirst_receiver_oracle_audit_llama3_2_3b_to_qwen3_4b_seed1234
nohup bash launch_when_cuda_ready.sh >/dev/null 2>&1 &
tail -f logs/pipeline.log
```

Machine-readable output is written to `runs/study/results/comparison.json` and appended to the
shared experiment-results document only after successful completion.
