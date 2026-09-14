# OpenBookQA: true Options-first old method vs posterior-Question 32-token method

Zero-shot paired evaluation on the same fixed seed-1234 sample of 128 English OpenBookQA test questions.

- Old method: `Candidate answers -> Question -> instruction -> Answer:`. The Qwen Receiver gets native token0 plus 32 external Candidate-answer KVs first, then processes Question natively.
- New method: standard raw order with Sender `Question -> Options -> repeated Question`; Receiver uses native Question, 32 external Options KVs, then native `Answer:`.
- Each method reuses its own MMLU-Pro-trained Full28-Diagonal best-validation-accuracy checkpoint.
- No OpenBookQA label is used for training or checkpoint selection.
- Intermediate KV caches are deleted after successful evaluation.

## Study result (128 samples)

| Condition | Accuracy | Oracle agreement | Oracle choice-KL |
|---|---:|---:|---:|
| Old Options-first Qwen full native | 50.78% | 100.00% | 0.000000 |
| Old Options-first native oracle | 47.66% | 100.00% | 0.000000 |
| Old Options-first translated | 28.12% | 42.97% | 0.577272 |
| New posterior-Question Qwen full native | 82.81% | 100.00% | 0.000000 |
| New posterior-Question native oracle | 75.00% | 100.00% | 0.000000 |
| New posterior-Question translated | 46.09% | 46.88% | 0.914871 |

Translated paired counts: both correct 24, only old correct 12, only new
correct 35, and both wrong 57. New minus old accuracy is +17.97 percentage
points (exact two-sided McNemar p=0.001089).
