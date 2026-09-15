# Forward train4096 results

The full experiment completed successfully on the official OpenBookQA Main splits with 4096 training, 128 validation, and 128 held-out test examples.

## Test results

| Method | Correct | Accuracy | Oracle agreement | Choice KL | K cosine | V cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-A Full28-MLP | 60/128 | 46.88% | 50.78% | 0.9496 | 0.9525 | 0.8006 |
| Shared-parameter Stage-B | 94/128 | 73.44% | 67.97% | 0.6115 | 0.8805 | 0.7769 |
| Frozen-base Residual64 Stage-B | **95/128** | **74.22%** | **70.31%** | **0.5615** | **0.9506** | **0.7993** |

## Training-volume comparison

| Method | Train1024 | Train4096 | Change |
| --- | ---: | ---: | ---: |
| Stage-A | 48.44% | 46.88% | -1.56 pp |
| Shared Stage-B | 69.53% | 73.44% | +3.91 pp |
| Residual64 Stage-B | 66.41% | **74.22%** | **+7.81 pp** |

Increasing the functional-training coverage materially improves both Stage-B methods. Residual64 benefits most and now slightly exceeds Shared Stage-B, while preserving substantially higher K/V similarity.

Shared and Residual64 have similar aggregate accuracy but different predictions: 87 examples are jointly correct, 7 only Shared, 8 only Residual64, and 26 jointly wrong. The paired McNemar p-value is 1.0, so the 1-example difference is not statistically meaningful on test128.

Residual64's mean relative correction norm is 5.98% for K and 8.47% for V. It trains 9,437,184 parameters, compared with 283,115,520 parameters for shared Stage-B.

Machine-readable metrics and all per-sample records are in `runs/study/results/`.
