# Quantized Cross-Model KV Translation

This directory contains the Qwen3-0.6B to Qwen3-1.7B FP16/INT4 control
experiment. Scripts are generated only; model loading, training, and evaluation
have not been run.

## Controls

The four groups are:

- `fp16_fp16`
- `int4_fp16`
- `fp16_int4`
- `int4_int4`

INT4 means weight-only bitsandbytes NF4 with FP16 compute. KV tensors remain in
the model compute dtype. The loader asserts that real 4-bit parameters exist and
never silently falls back to FP16.

The current `attnkv` environment does not contain bitsandbytes. Install and pin a
compatible version before any INT4 stage. Run `preflight` again after installing.

## Metric Boundaries

The untrained representation stage reports relation geometry:

- K/V cosine only when tensor structures match exactly
- K/V token-token Gram correlation
- K/V RSA
- K/V kNN overlap
- K/V joint-geometry consistency
- sender-native versus receiver-native self-attention route similarity

It does not report receiver readability or attention output similarity.

Translator evaluation always uses the native cache from the exact same receiver
as its reference. Every record includes:

- `receiver_native_ce`
- `receiver_native_logit_quality`
- `receiver_native_f1`
- `translated_ce`
- `ce_delta`
- `translated_f1`
- `top1_match`
- `attention_output_cos`

Do not compare absolute translated CE directly between FP16 and INT4 receivers.
Use `ce_delta`, same-receiver `top1_match`, and same-receiver
`attention_output_cos`.

## Structure Handling

The scripts inspect layers, KV heads, head dimension, and RoPE configuration.
For the current local Qwen3 pair they use direct layer/head comparison:

```text
layers=28, kv_heads=8, head_dim=128, rope_theta=1000000
```

If a future pair differs, relation metrics use fixed proportional layer/head
mapping and direct tensor cosine is disabled.

## Explicit Stages

```bash
bash runs/quantized_real_kv_translation_seed1234/run_all.sh preflight

bash runs/quantized_real_kv_translation_seed1234/run_all.sh representation fp16_fp16
bash runs/quantized_real_kv_translation_seed1234/run_all.sh representation int4_fp16
bash runs/quantized_real_kv_translation_seed1234/run_all.sh representation fp16_int4
bash runs/quantized_real_kv_translation_seed1234/run_all.sh representation int4_int4

bash runs/quantized_real_kv_translation_seed1234/run_all.sh drift qwen3_0_6b
bash runs/quantized_real_kv_translation_seed1234/run_all.sh drift qwen3_1_7b

bash runs/quantized_real_kv_translation_seed1234/run_all.sh train_mse_then_ce GROUP
bash runs/quantized_real_kv_translation_seed1234/run_all.sh train_ce_only_small GROUP
bash runs/quantized_real_kv_translation_seed1234/run_all.sh eval GROUP mse_then_ce
bash runs/quantized_real_kv_translation_seed1234/run_all.sh eval GROUP ce_only_small

bash runs/quantized_real_kv_translation_seed1234/run_all.sh package
```

Replace `GROUP` with one of the four group names. The runner deliberately has no
default command that launches all expensive jobs.

Packaged tables will be written to:

- `summary/representation_comparison.csv`
- `summary/translation_comparison.csv`
- `summary/same_model_drift.json`
- `summary/SUCCESS.json`
