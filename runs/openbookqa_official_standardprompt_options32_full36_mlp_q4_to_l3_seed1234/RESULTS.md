# Reverse Qwen3-4B to Llama3.2-3B results

The complete study finished successfully on the official OpenBookQA Main splits with 1024 training, 128 validation, and 128 held-out test examples.

## Test results

| Method | Accuracy | Oracle agreement | Choice KL | K cosine | V cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage-A Full36-MLP | 60.94% | 72.66% | 0.2046 | 0.9547 | 0.7638 |
| Shared-parameter Stage-B | **62.50%** | 64.84% | 0.2418 | 0.8853 | 0.7238 |
| Frozen-base Residual64 Stage-B | **62.50%** | **69.53%** | **0.2086** | **0.9539** | **0.7637** |

The Llama Native Oracle32 accuracy for the same selected positions is 61.72% on validation. On test, Stage-A reaches 60.94%, while both Stage-B variants reach 62.50%.

Shared and Residual64 have identical aggregate test accuracy but different per-sample predictions: 69 examples are jointly correct, 11 are correct only for Shared, 11 only for Residual64, and 37 are jointly wrong.

Residual64 preserves the Stage-A representation substantially better than shared fine-tuning. Its mean relative correction norm is 4.23% for K and 1.65% for V, while retaining K/V cosine values of 0.9539/0.7637. It trains 7,340,032 parameters versus 278,921,216 for shared Stage-B.

The complete machine-readable metrics and all 128 per-sample records are in `runs/study/results/`.
