# Frozen Writer cross-dataset results

All OpenBookQA-trained Stage-A Writers and Stage-B Residuals were frozen.  Neither
ARC-Challenge nor MMLU-Pro was used for training, checkpoint selection, or adaptation.
Each test uses 128 examples selected with seed 1234 after an OpenBookQA overlap audit.

## ARC-Challenge

| Receiver / route | Native or Oracle | Stage-A | Frozen Residual |
| --- | ---: | ---: | ---: |
| Qwen full native | **85.94%** | - | - |
| Llama -> Qwen | 85.16% Oracle32 | 54.69% | **64.84%** |
| Gemma -> Qwen | 84.38% Oracle32 | 52.34% | **69.53%** |
| Gemma full native | 76.56% | - | - |
| Qwen -> Gemma | 73.44% Oracle32 | 75.78% | **78.12% Mixed-Hub** |
| Llama -> Qwen -> Gemma | 73.44% Oracle32 | 51.56% | **57.03% Mixed-Hub** |
| Llama full native | **71.88%** | - | - |
| Qwen -> Llama | 57.03% Oracle32 | 58.59% | **63.28%** |
| Gemma -> Qwen -> Llama | 60.16% Oracle32 | 57.03% | **57.81% Llama residual** |

The frozen functional Residual improves every one-hop route on ARC-Challenge.  The
Mixed-Hub Gemma Residual also improves the canonical forward two-hop route from 51.56%
to 57.03%, although a substantial gap to the 73.44% same-anchor Oracle remains.

## MMLU-Pro

| Receiver / route | Native or Oracle | Stage-A | Frozen Residual |
| --- | ---: | ---: | ---: |
| Qwen full native | **45.31%** | - | - |
| Llama -> Qwen | 37.50% Oracle32 | 13.28% | **17.97%** |
| Gemma -> Qwen | 39.84% Oracle32 | 14.84% | **24.22%** |
| Gemma full native | **31.25%** | - | - |
| Qwen -> Gemma | 27.34% Oracle32 | **21.09%** | 20.31% Mixed-Hub |
| Llama -> Qwen -> Gemma | 27.34% Oracle32 | 15.62% | **17.97% Mixed-Hub** |
| Llama full native | **32.03%** | - | - |
| Qwen -> Llama | 23.44% Oracle32 | 21.09% | **22.66%** |
| Gemma -> Qwen -> Llama | 25.78% Oracle32 | 19.53% | **20.31% Llama residual** |

MMLU-Pro is a much stronger distribution and answer-space shift (4--10 choices in the
selected subset).  The frozen Writers remain above the receiver no-memory baselines, and
most Residuals recover part of the Stage-A gap, but the Writer-to-Oracle gaps are large.
The Qwen-to-Gemma Mixed-Hub Residual does not improve the one-hop Stage-A result on this
dataset (20.31% versus 21.09%).

## Dataset audit

- ARC-Challenge: 128 four-choice examples selected from 1,144 eligible test examples;
  maximum question-token Jaccard against the 4,096 OpenBookQA training questions was 0.875.
- MMLU-Pro: 128 category-balanced examples selected from 12,032 test examples; 96 have ten
  choices and the remainder have four to nine; maximum Jaccard was 0.600.
- No selected example reached the 0.9 near-duplicate exclusion threshold.

Complete choice logits, per-example predictions, representation metrics, Native/Oracle
agreement, common/only-correct counts, and McNemar tests are stored under each dataset's
`results/` directory.
