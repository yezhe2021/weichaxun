# P1-Causal: receiver-native two-hop evidence control

This experiment asks whether a trainable reader/writer can causally control a fully frozen Qwen3-8B receiver when evidence is complete, token-level, and encoded by the same Qwen3-8B checkpoint. It does not use heterogeneous senders, native KV, token compression, head routing, or layer routing.

## Data

The generator covers 12 typed relation schemas, including employment, education, film, books, research, music, products, events, object location, habitats, routes, and organizations. Every example has the same causal structure:

```text
x --r1--> bridge --r2--> answer
```

Source A identifies the bridge. Source B contains four bridge-to-answer candidates. Every question has a paired counterfactual variant in which only the target B relation changes. Test examples use held-out entity assignments and held-out paraphrase templates.

## Pipeline

Run the stages separately:

```bash
cd /home/yezhe/伪查询

bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh generate-data
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh full-text-gate
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh benchmark-all
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh cache-train
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh cache-test
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh train
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh eval
```

`full-text-gate` must be inspected before caching or training. If the frozen receiver cannot solve the direct full-text task and follow counterfactual changes, adapter results are not interpretable.

`benchmark-all` runs six evaluation variants in one model-loading session: the original direct baseline, improved formatting and extraction, Qwen3 thinking mode, two-shot chain-of-thought, candidate sequence ranking, and five-path self-consistency. Results are written under `full_text_benchmark_qwen3_8b`.

The Qwen3-8B artifacts are isolated under `full_text_gate_qwen3_8b`, `cache_qwen3_8b`, `train_qwen3_8b`, and `eval_qwen3_8b` so they cannot be mixed with earlier Qwen3-1.7B or Qwen3-4B outputs. Native memory is captured at layer 18 and writers are inserted at layers 12, 24, and 34.

The cache stores all evidence-token hidden states from receiver layer 14, the aligned question state, and exact candidate span masks. It is sharded to avoid loading the complete token memory into RAM or GPU at once.

The trainable adapter contains:

- a first cross-attention read over A;
- a second cross-attention read over B;
- bridge and answer pointer objectives;
- independent norm-calibrated residual writers at receiver layers 8, 16, and 24;
- bounded gates;
- answer and single-source rejection losses.

The receiver parameters are frozen throughout.

## Evaluation

Free-running evaluation reports:

- question-only, A-only, B-only, zero, correct, mismatched, and state-swap EM;
- paired counterfactual consistency;
- bridge pointer accuracy;
- answer pointer accuracy from `s1` and `s2`;
- candidate sequence log-probability shifts;
- macro and per-schema performance;
- worst-schema correct EM.

State swap is performed within a counterfactual pair, keeping the question and A fixed while replacing only the composed reader state.

## Smoke configuration

Use separate output roots so formal artifacts are not overwritten:

```bash
TRAIN_PAIRS=32 VALID_PAIRS=8 TEST_PAIRS=16 \
DATA_ROOT=runs/composable_evidence_kv_p1_causal_native_seed1234/data_smoke \
bash runs/composable_evidence_kv_p1_causal_native_seed1234/run_all.sh generate-data
```

Apply matching `DATA_ROOT`, `CACHE_ROOT`, `TRAIN_OUT`, and `EVAL_OUT` overrides to each following stage. The scripts do not automatically run subsequent stages.
