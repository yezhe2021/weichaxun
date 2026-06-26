# Two-Stage Heterogeneous KV Diagnosis

This experiment diagnoses Qwen3-0.6B -> Qwen3-1.7B KV translation failures without training a new translator.

It reuses the existing real heterogeneous translator:

`runs/real_qwen3_0_6b_to_1_7b_seed1234/real_kv_translator.py`

and evaluates trained checkpoints, primarily `mse_then_ce`, with optional `mse_only` as a control.

## Stage 1: Coherent Cache Swap

For each sample:

1. Sender Qwen3-0.6B prefills context `C` only.
2. Receiver Qwen3-1.7B prefills context `C` only.
3. `RealCrossModelKVTranslator` maps sender context KV into receiver-shaped translated context KV.
4. Receiver does not re-input `C`; it continues teacher-forcing on `Q + answer_prefix` from one of these context caches:
   - `native`
   - `translated`
   - `native_K + translated_V`
   - `translated_K + native_V`

Metrics:

`ce_delta`, `logit_kl`, `top1_match`, `answer_f1`, `attention_output_cos`, `kv_joint_consistency`.

Interpretation:

- `native_K + translated_V` bad: translated V content is unreadable.
- `translated_K + native_V` bad: translated K routing is wrong.
- both swaps good but full translated bad: K/V pairing is inconsistent.

## Stage 2: Receiver-Q Readout Probe

This is offline only. It does not replace the model attention matrix and does not feed probe outputs back into the receiver.

For each sample:

1. Run receiver native context cache + teacher-forced `Q + answer_prefix`.
2. Record each layer's native rotated query states at answer-prediction positions.
3. Offline compute how those fixed receiver-native Q states read:
   - `native_K + native_V`
   - `translated_K + translated_V`
   - `native_K + translated_V`
   - `translated_K + native_V`

Metrics:

`route_overlap`, `attention_js`, `attention_output_cos`, `output_mse`.

## Run

Run only the primary checkpoint:

```bash
cd /home/yezhe/伪查询
bash runs/two_stage_heterogeneous_kv_diagnosis_seed1234/run_all.sh diagnose_mse_then_ce
bash runs/two_stage_heterogeneous_kv_diagnosis_seed1234/run_all.sh package
```

Run the control checkpoint:

```bash
cd /home/yezhe/伪查询
bash runs/two_stage_heterogeneous_kv_diagnosis_seed1234/run_all.sh diagnose_mse_only
bash runs/two_stage_heterogeneous_kv_diagnosis_seed1234/run_all.sh package
```

Outputs are written under:

`runs/two_stage_heterogeneous_kv_diagnosis_seed1234/results/<checkpoint_label>/`

and packaged summaries under:

`runs/two_stage_heterogeneous_kv_diagnosis_seed1234/summary/`
