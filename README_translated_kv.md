# Translated-like KV diagnostics

This experiment perturbs or reconstructs one receiver model's real post-RoPE
`DynamicCache`. It does not claim to implement heterogeneous KV translation.

## Protocol

The prompt is split into context `C` and `question + Answer:` suffix `Q`.
The native reference prefills `C`, then processes `Q` with the native cache.
Every translated condition changes only the context cache. Query and answer
tokens are processed natively by the frozen receiver.

Before accepting a run, inspect `summary.json/equivalence`. One-shot `C+Q` and
split `C -> cache -> Q` must pass the configured logit tolerance with complete
top-1 agreement.

## Deterministic controls

```bash
conda run -n attnkv python translated_kv_diagnostics.py \
  --device cuda --dtype float16 \
  --max-samples 64 --max-context-tokens 256 \
  --methods native,noise,token_shuffle,head_shuffle,low_rank,rope_shift \
  --residual-alphas 0.25,0.5,0.75,1.0 \
  --out runs/translated_kv_controls_n64
```

Use `--device cpu --dtype float32` for CPU runs. Full training is substantially
slower on CPU. V100 runs must use float16 rather than bfloat16.

## Learned translators

All objective comparisons use the same fixed compressed pseudo-sender input and
decoder architecture. Only the loss changes.

```bash
conda run -n attnkv python train_kv_translator.py \
  --objective mse --max-train-samples 512 --max-val-samples 64 \
  --out runs/translator_mse

conda run -n attnkv python train_kv_translator.py \
  --objective ce --max-train-samples 512 --max-val-samples 64 \
  --out runs/translator_ce

conda run -n attnkv python train_kv_translator.py \
  --objective mse_ce --max-train-samples 512 --max-val-samples 64 \
  --out runs/translator_mse_ce
```

The joint nonlinear autoencoder control trains its encoder as well as decoder:

```bash
conda run -n attnkv python train_kv_translator.py \
  --translator-kind autoencoder --objective mse \
  --out runs/translator_autoencoder
```

Add `--rope-disentangled` to compare inverse-RoPE translation followed by exact
receiver-side RoPE restoration.

Evaluate a checkpoint, including direct replacement and residual fusion:

```bash
conda run -n attnkv python translated_kv_diagnostics.py \
  --methods native,translator \
  --translator-checkpoint runs/translator_ce/checkpoint_epoch1.pt \
  --residual-alphas 0.25,0.5,0.75,1.0 \
  --out runs/translator_ce_diagnostics
```

## Outputs

- `summary.json`: arguments, equivalence guard, and aggregate diagnostic table.
- `diagnostic_table.csv`: requested comparison table with bootstrap intervals.
- `per_example.jsonl`: generation and functional metrics for every sample.
- `per_layer.jsonl`: cache and attention metrics by layer and suffix scope.

Core fields are `kv_mse`, `kv_relative_mse`, `k_cos`, `v_cos`, `logit_kl`,
`ce_delta`, `top1_match`, `attention_route_overlap`, `attention_route_js`,
`attention_output_cos`, and `kv_joint_consistency`.

## Formal seed-1234 run

The completed Qwen3-0.6B formal run uses 512 HotpotQA training examples and 64
development examples. Its consolidated report and tables are under:

```text
runs/formal_translated_kv_summary_seed1234/
```

Raw per-layer files and translator checkpoints remain on the experiment server.
They are intentionally excluded from Git because the checkpoints exceed normal
GitHub file-size limits. The repository contains the report, aggregate tables,
training histories, metadata, and sample-level evaluation results.
