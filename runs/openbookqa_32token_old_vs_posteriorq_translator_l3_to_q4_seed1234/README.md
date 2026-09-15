# OpenBookQA 32-token old vs posterior-Question Translator

Zero-shot transfer audit on a fixed seed-1234 sample of 128 English OpenBookQA test questions.

- Old: route 32 Options regions with the final `Answer:` token.
- New: route the same 32 Options regions with the last token of a repeated posterior Question.
- Both reuse their own MMLU-Pro-trained Full28-Diagonal checkpoint; no OpenBookQA labels are used for training.
- Receiver protocol is held fixed: native Question, 32 external Options KV, native `Answer:`.
- Results include per-sample predictions and paired correctness counts.

Run with `/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study`.

## Study result (128 samples)

| Condition | Accuracy | Oracle agreement | Oracle choice-KL |
|---|---:|---:|---:|
| Qwen full native | 82.81% | 100.00% | 0.000000 |
| Old native oracle | 77.34% | 100.00% | 0.000000 |
| Old translated | 51.56% | 46.09% | 1.080689 |
| New native oracle | 75.00% | 100.00% | 0.000000 |
| New translated | 46.09% | 46.88% | 0.914871 |

Old-vs-new translated paired counts: both correct 39, only old correct 27,
only new correct 20, both wrong 42. The new-minus-old accuracy difference is
-5.47 percentage points (exact two-sided McNemar p=0.381693).
