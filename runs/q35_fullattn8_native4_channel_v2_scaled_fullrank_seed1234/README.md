# Qwen3.5 full-attention Sender v2: scaled full-rank

This independent experiment preserves the previous Q35 result as a baseline and
tests whether per-feature train-only RMS scaling plus independent full-rank
K/V basis maps can register Qwen3.5's eight full-attention layers into the
frozen Qwen3-4B native KV channel.

Pipeline:

1. Rebuild context-only Qwen3.5 caches and detailed tokenizer alignment audit.
2. Compute fixed source `[8,1024]` and target `[36,1024]` RMS scales.
3. Evaluate V0 scaled deterministic interpolation.
4. Run a 16-sample A1 overfit diagnostic.
5. A1 trains only eight independent full-rank K and V feature maps.
6. A2 trains K/V-specific depth maps and target calibration while slowly
   updating feature maps.
7. B trains the Writer with gold-answer teacher-forcing cross-entropy only.
8. Unified evaluation includes native controls, prior Writer8, old Q35 S2,
   V0/V1/V2, shuffled and forced-zero controls.

No stage uses a hard pass/fail gate.

The complete per-token offset/alignment metadata is stored as
`alignment_metadata.json.gz` because the uncompressed development audit is
close to GitHub's single-file size limit.
