# P3-B Multi-Layer Question-Independent Evidence-KV Readability Diagnosis

P3-B tests whether frozen Qwen3-8B Native states preserve reusable HotpotQA evidence when the Sender never sees the question. The main `evidence_only` branch encodes only the ordered official gold supporting evidence. `question_evidence` is an explicitly labeled task-conditioned upper bound.

## Contract

- Extractive HotpotQA examples only; yes/no and non-extractive examples are saved separately.
- Frozen Qwen3-8B question encoder and Sender.
- 36 layers, all evidence tokens, 8 KV heads flattened to 1024 dimensions per token.
- Pre-RoPE K, native V, and complete layer hidden states are cached in FP16.
- K/V retain the same layer and token index. There is no layer averaging or slot pooling.
- Layer-independent PCA, random, and trainable 1024-to-256 mappings.
- Question-conditioned soft layer routing over `last1`, `last4`, `last8`, `uniform16`, and `all36`.
- Start/end span prediction only. No LM head and no free-running generation.

## Controls

Each configuration reports correct, shuffled, zero, question-only, K/V mismatch, synchronized token permutation, and layer permutation. Predictions, start/end accuracy, current-answer EM/F1, source-memory EM/F1, supporting-fact recall, losses, and layer weights are saved.

Cross-sample shuffled evidence creates a deliberate semantic mismatch: the current question does not specify which span was the answer to the source sample's different question. Source-answer hit rate is therefore reported as a causal diagnostic, but it is not used as the validity gate. The validity gate instead requires an evidence-only 16-sample branch to overfit correct spans and substantially beat zero memory.

## Execution

Caching is sequential because the server has one 32 GiB V100. After Qwen3-8B exits, independent probe families run two at a time by default. Set `P3B_PARALLEL_JOBS` to change this.

```bash
bash /home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234/run_all.sh all
```

Status:

```bash
bash /home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234/run_all.sh status
```
