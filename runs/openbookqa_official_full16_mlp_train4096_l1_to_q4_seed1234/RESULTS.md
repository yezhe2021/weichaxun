# Results: Llama3.2-1B -> Qwen3-4B

## Protocol

- Dataset: official OpenBookQA Main, deterministic `4096/128/128` train/validation/test subsets, seed `1234`.
- Sender / receiver: Llama3.2-1B / Qwen3-4B.
- Memory: RawAnchor32 selected option-token KV; the receiver keeps its native question representation.
- Stage A: independent per-target-layer/per-head K/V MLP, `16 x 8 x 64 -> 36 x 8 x 128`, 2,048 optimizer steps.
- Stage B: freeze Stage A and train a rank-64 receiver-space residual adapter for 1,024 optimizer steps.

## Test results (128 examples)

| Condition | Correct | Accuracy |
|---|---:|---:|
| Qwen full native | 96 | 75.00% |
| Qwen native Oracle32 at Llama1-selected positions | 95 | 74.22% |
| Residual Stage B | 73 | 57.03% |
| Llama1 full native | 70 | 54.69% |
| Llama1 native selected32 | 58 | 45.31% |
| Stage A | 38 | 29.69% |
| Qwen shuffled Oracle32 | 34 | 26.56% |
| Qwen no memory | 33 | 25.78% |
| Qwen zero32 | 31 | 24.22% |

| Translator metric | Stage A | Residual Stage B |
|---|---:|---:|
| Accuracy | 29.69% | 57.03% |
| Oracle agreement | 30.47% | 60.16% |
| Choice KL | 1.242756 | 0.764105 |
| K cosine | 0.943161 | 0.941092 |
| V cosine | 0.762891 | 0.761607 |

Residual relative norms were `0.048858` for K and `0.070717` for V. Stage A has 94,371,840 trainable parameters; the residual adapter has 9,437,184. Paired correctness counts were: both correct 28, only Stage A 10, only residual 45, both wrong 45 (`p=2.057e-6`). The residual improves accuracy by 27.34 percentage points while preserving Stage-A similarity, but remains 17.19 points below the native Oracle32.

## Data caveat

The official subsets contain one semantic train/test duplicate under option permutation (`test 9-317`, `train 13-308`). The residual and all native baselines answer this test item correctly, while Stage A does not. Excluding it gives residual accuracy `72/127 = 56.69%`; the main table preserves the predefined 128-item protocol for comparability.
