# OpenBookQA standard-prompt Full28-MLP

Mainline experiment under the frozen protocol:

```text
Question -> Options -> Answer:
```

- Dataset: AraDiCE OpenBookQA English, 497 examples.
- Split: seed1234 fixed test128 used by the linear zero-shot baseline; the disjoint remainder is shuffled into train305 and validation64.
- Sender: Llama-3.2-3B; Receiver: Qwen3-4B.
- Memory: 32 Options-only RawAnchor KVs; Qwen-native token0 and Question KV are retained.
- Translator: Full28 MLP, independent K/V and target layers, no token/head mixing and no bias. Each depth map is `3584 -> 1024 -> 128` with GELU, followed by independent diagonal head adapters.
- Stage A: KV reconstruction, 8 true epochs (312 optimizer steps).
- Stage B: final-position A-D choice-only Oracle KL, 4 true epochs (156 optimizer steps).
- Checkpoint selection uses validation only; the fixed test128 is evaluated only at the end.

Intermediate caches and `.pt` checkpoints are not intended for upload.

## Study result

| Condition | Accuracy | Oracle agreement | Oracle choice-KL |
|---|---:|---:|---:|
| Qwen Full Native | 82.81% | 91.41% | 0.113281 |
| Llama Full Native | 75.00% | 66.41% | 0.739892 |
| Qwen Self-Selected32 Native | 77.34% | 95.31% | 0.020482 |
| Llama-selected Qwen Native Oracle | 77.34% | 100.00% | 0.000000 |
| Full28-MLP | 56.25% | 60.94% | 0.906609 |
| Shuffle | 23.44% | 25.00% | 2.039195 |
| Zero | 31.25% | 32.81% | 1.855395 |
| No memory | 33.59% | 35.94% | 2.435758 |

The matching zero-shot Full28-Diagonal Linear baseline is 51.56%; MLP improves
accuracy by 4.69 percentage points. Paired counts are: both correct 44, only
Linear correct 22, only MLP correct 28, both wrong 34. The exact two-sided
McNemar p-value is 0.479888, so this 128-sample improvement is not statistically
significant.
