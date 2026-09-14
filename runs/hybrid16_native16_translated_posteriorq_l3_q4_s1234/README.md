# Hybrid 16 Native + 16 Translated KV Audit

This is an evaluation-only diagnostic based on the completed posterior-Question
Llama-3.2-3B to Qwen3-4B Full28-Diagonal experiment.

It preserves the same 32 selected option anchors, cache length, position IDs,
attention mask, receiver layout, and trained Translator. Exactly 16 memory
positions use position-matched native Qwen KV while the other 16 retain the
translated Llama KV. K and V are always replaced together.

Two complementary patterns are evaluated for both validation-selected Stage-B
checkpoints:

- native even indices, translated odd indices;
- translated even indices, native odd indices.

No training is performed.

Run:

```bash
bash launch.sh
tail -f logs/pipeline.log
```

Final results are written to `runs/study/results/comparison.json`.
