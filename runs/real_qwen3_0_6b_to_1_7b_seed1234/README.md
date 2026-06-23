# Real Qwen3 0.6B to 1.7B Context KV Translation

This folder contains the real heterogeneous KV translation experiment.

Main experiment definition:

```text
sender Qwen3-0.6B prefill(C) -> sender context KV
receiver Qwen3-1.7B prefill(C) -> receiver native context KV
translator(sender context KV) -> receiver-shaped translated context KV
receiver uses translated context KV as past_key_values
receiver natively processes Q + answer teacher forcing
```

The translator never receives or translates query KV in the main experiment. `C+Q+A` full-forward equivalence is used only as a sanity check for receiver cache continuation.

## Files

```text
real_kv_common.py              Shared tokenizer checks, RoPE utilities, metrics, cache helpers.
real_kv_translator.py          Structured per-layer/per-KV-head translator with RoPE strip/restore and learnable gates.
train_real_kv_translator.py    Training entrypoint for MSE-only, CE-only, and MSE+CE.
eval_real_kv_translation.py    Evaluation entrypoint for diagnostic metrics.
package_results.py             Result completeness checker and summary aggregator.
run_all.sh                     Reproducible command wrapper.
```

## Training Groups

```text
mse_only       Fit receiver native context KV only.
ce_only        Optimize answer-token teacher-forcing CE only.
mse_ce         Joint KV MSE and answer CE.
mse_then_ce    MSE pretrain, then lower-learning-rate CE fine-tune.
```

## Expected Outputs

```text
train/<group>/checkpoint_epoch1.pt
train/<group>/metadata.json
train/<group>/train_history.jsonl
eval/<group>/summary.json
eval/<group>/diagnostic_table.csv
eval/<group>/per_example.jsonl
eval/<group>/per_layer.jsonl
summary/all_diagnostic_rows.csv
summary/SUCCESS.json
```

## Commands

From `/home/yezhe/伪查询`:

```bash
bash runs/real_qwen3_0_6b_to_1_7b_seed1234/run_all.sh train
bash runs/real_qwen3_0_6b_to_1_7b_seed1234/run_all.sh eval
bash runs/real_qwen3_0_6b_to_1_7b_seed1234/run_all.sh package
```

Run one stage at a time if you want easier process control:

```bash
bash runs/real_qwen3_0_6b_to_1_7b_seed1234/run_all.sh train_mse_only
bash runs/real_qwen3_0_6b_to_1_7b_seed1234/run_all.sh eval_mse_only
```

No result interpretation is encoded in this folder. The scripts only produce metrics and completeness metadata.
