# Results

The final evaluation completed on 128 held-out OpenBookQA examples.  The checkpoint was
selected only from the validation split; the selected checkpoint is step 512.

| Hub input | Stage-A | Previous Stage-B | Mixed-Hub Stage-B |
| --- | ---: | ---: | ---: |
| Qwen native Hub | 70.31% (90/128) | 71.09% (91/128) | **73.44% (94/128)** |
| Llama Stage-A Hub | 61.72% (79/128) | 61.72% (79/128) | **64.06% (82/128)** |

The source-agnostic Mixed-Hub Residual64 improves the Qwen-native input by 3.13 percentage
points over Stage-A and the Llama Stage-A input by 2.34 points.  Thus the B0 feasibility
criterion is met on this split: improving the translated Llama Hub does not require degrading
the native Qwen Hub.

The paired changes are small on 128 examples and are not statistically conclusive
(McNemar p=0.21875 for Qwen Stage-A vs mixed and p=0.60724 for Llama Stage-A vs mixed).
Complete metrics and per-example predictions are stored under `results/`.
