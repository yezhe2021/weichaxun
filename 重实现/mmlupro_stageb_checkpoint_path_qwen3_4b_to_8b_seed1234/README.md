# 4B→8B Stage-B checkpoint-path comparison

This diagnostic reuses the existing Full36+Head8 Stage-A checkpoint and
reruns Stage B for exactly four epochs without early stopping. Every epoch is
saved. Formal test evaluation compares Stage A, the lowest-validation Stage-B
epoch, and the final Stage-B epoch using final full-vocabulary KL, choice-only
KL, centered choice-logit MSE, Native agreement, and accuracy.
