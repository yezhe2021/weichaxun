# P3-E-G Strong Reader V1

This experiment keeps the existing Qwen3-8B Sender, C2 learned Writer,
Canonical memory, C1 Query Adapter/head routing, Qwen3-4B backbone, and native
`o_proj` frozen. It changes only the post-`o_proj` output conversion and
injection control.

## Correct Qwen3-4B dimensions

The 32 x 128 headwise readout is flattened to 4096 dimensions and passed through
the frozen native `o_proj: 4096 -> 2560`. Therefore the requested post-`o_proj`
adapter is implemented as:

```text
RMSNorm(2560) -> Linear(2560,128) -> SiLU -> Linear(128,2560)
```

Its up projection is zero-initialized. The adapted residual is the original
`o_proj` output plus this low-rank update.

The token-conditioned gate is:

```text
RMSNorm(receiver attention input, 2560) -> Linear(2560,1) -> sigmoid
```

Gate weights are initialized to zero and each bias is initialized to the logit
of the corresponding old C1 scalar gate. At step zero, Strong Reader output is
therefore equivalent to C1 output.

## Training

- Smoke: 16 rows, 5 epochs.
- Formal: the existing 512-row cache, independently initialized from C1, 5
  epochs.
- Loss: `L_answer + 0.5 * relu(0.5 + NLL_correct - NLL_shuffled)`.
- Only the new output adapter, its RMSNorm, token-gate RMSNorm, and token-gate
  linear layer are optimized.
- Validation64 is used only after formal training.

## Evaluation

Conditions:

- `question_only`
- `old_current_reader`
- `strong_reader_v1`
- `hard_shuffled_strong_reader`
- `oracle_support_strong_reader`
- `reader_off`

Automatic EM/F1, bridge/comparison scores, EOS, memory-switch behavior, token
gate distributions, adapter norm ratios, cosine diagnostics, and per-example
outputs are saved. A blinded manual C/P/W worksheet is generated without
fabricating semantic labels.

## Run

```bash
bash /home/yezhe/伪查询/runs/p3e_g_strong_reader_v1_seed1234/launch.sh
tail -f /home/yezhe/伪查询/runs/p3e_g_strong_reader_v1_seed1234/p3e_g_run.log
```
