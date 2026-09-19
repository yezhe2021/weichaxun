# Results: Qwen3-4B -> Llama3.2-1B

## Protocol

- Dataset: official OpenBookQA Main, deterministic `4096/128/128` train/validation/test subsets, seed `1234`.
- Sender / receiver: Qwen3-4B / Llama3.2-1B.
- Memory: RawAnchor32 selected option-token KV; the receiver keeps its native question representation.
- Stage A: independent per-target-layer/per-head K/V MLP, `36 x 8 x 128 -> 16 x 8 x 64`, 2,048 optimizer steps.
- Stage B: freeze Stage A and train a rank-64 receiver-space residual adapter for 1,024 optimizer steps.

## Test results (128 examples)

| Condition | Correct | Accuracy |
|---|---:|---:|
| Qwen full native | 96 | 75.00% |
| Qwen native selected32 | 94 | 73.44% |
| Llama1 full native | 70 | 54.69% |
| Residual Stage B | 59 | 46.09% |
| Stage A | 57 | 44.53% |
| Llama1 native Oracle32 at Qwen-selected positions | 52 | 40.63% |
| Llama1 zero32 | 34 | 26.56% |
| Llama1 no memory | 33 | 25.78% |
| Llama1 shuffled Oracle32 | 29 | 22.66% |

| Translator metric | Stage A | Residual Stage B |
|---|---:|---:|
| Accuracy | 44.53% | 46.09% |
| Oracle agreement | 61.72% | 65.63% |
| Choice KL | 0.222369 | 0.188169 |
| K cosine | 0.977452 | 0.977165 |
| V cosine | 0.835195 | 0.835130 |

Residual relative norms were `0.025175` for K and `0.013857` for V. Stage A has 154,140,672 trainable parameters; the residual adapter has 2,097,152. Paired correctness counts were: both correct 53, only Stage A 4, only residual 6, both wrong 65 (`p=0.753906`). The residual gains 1.56 percentage points and exceeds the position-matched Llama native Oracle32 by 5.47 points, so that Oracle is not a strict upper bound for learned translated states.

## Data caveat

The official subsets contain one semantic train/test duplicate under option permutation (`test 9-317`, `train 13-308`). Both Stage A and residual answer this item correctly. Excluding it gives Stage-A accuracy `56/127 = 44.09%` and residual accuracy `58/127 = 45.67%`; the main table preserves the predefined 128-item protocol for comparability.
