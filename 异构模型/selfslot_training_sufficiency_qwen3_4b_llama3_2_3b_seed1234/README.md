# Self-Slot training sufficiency study

This experiment isolates whether the weak 64-slot self-model results were caused by only
32 optimizer steps and 256 sample exposures. Qwen3-4B and Llama-3.2-3B are studied
independently. Cross-family registration is not run.

All arms keep the original 64-slot compressor, pre-RoPE pooling, position policy,
final-position full-vocabulary native-to-slot KL, learning rate, batch size and seed.
Each arm uses exactly 512 optimizer steps and 4096 sample exposures:

| arm | unique train examples | epochs | validation |
| --- | ---: | ---: | --- |
| overfit16 | 16 | 256 | same 16 examples |
| repeat128 | 128 | 32 | disjoint 128 examples |
| diverse1024 | 1024 | 4 | same disjoint 128 examples |

All three arms of a model load byte-identical initial Slot queries. Sampling is a true
per-epoch permutation. Checkpoints and validation KL are saved every 64 optimizer steps.
Gold labels are never used in training or checkpoint selection.

Evaluation records full-vocabulary KL, choice-only KL, native choice agreement, accuracy,
mean off-diagonal slot K/V cosine, normalized attention entropy, effective attended tokens
and maximum attention weight. The overfit arm reports fit-set metrics separately from its
held-out test metrics. The other arms use the common validation and test sets.

Interpretation:

- Strong overfit16 fit proves the implementation can optimize this bottleneck on fixed data.
- repeat128 improvement isolates additional optimization/repetition.
- diverse1024 improvement over repeat128 at equal steps and exposures isolates data diversity.
- Failure of all three, especially overfit16, points to the compressor, position protocol,
  or final-position supervision rather than insufficient ordinary training.

Run:

```bash
cd /home/yezhe/异构模型/selfslot_training_sufficiency_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/study/status.json
```
