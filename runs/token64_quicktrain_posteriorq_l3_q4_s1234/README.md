# Token64 Full28-Diagonal Quick-Training Pilot

This experiment trains a new Llama-3.2-3B to Qwen3-4B Translator specifically
for 64 selected Options KV slots under the posterior-Question Sender protocol.

To obtain a fast diagnostic result, the budget is reduced to 512 train / 128
validation / 128 test samples, 256 Stage-A KV-reconstruction steps, and 128
Stage-B choice-KL steps. Checkpoint selection remains validation-only.

The Full28-Diagonal architecture is unchanged except that its token dimension is
now explicitly variable rather than guarded as exactly 32. Target layers and K/V
maps remain independent; heads and tokens never mix; all linear maps have no
bias.

The final evaluation automatically records paired correctness counts and sample
IDs for Translator64 versus Native64 Oracle, Translator64 versus the previous
Translator32, and Native64 Oracle versus Native32 Oracle.

```bash
bash launch.sh
tail -f logs/pipeline.log
```

Final results are saved under `runs/study/results/` and automatically appended
to `/home/yezhe/异构模型/EXPERIMENT_RESULTS.md` after successful completion.
