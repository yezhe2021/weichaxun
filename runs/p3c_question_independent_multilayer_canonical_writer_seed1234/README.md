# P3-C Question-Independent Multi-Layer Canonical Writer Bootstrap

P3-C trains a sender-specific, question-independent Qwen3-8B Writer that maps P3-B Native Evidence KV into a frozen public representation with shape `G x T_E x 256`. It preserves every evidence token and either all 36 layers or the fixed uniform 16-layer subset.

## Isolation contract

- The Sender cache is P3-B `evidence_only`; the Sender never receives the question.
- Qwen3-8B, the independent question encoder, and the best P3-B Native span teacher remain frozen.
- Every original layer has independent K and V 1024-to-256 projections and independent rank-32 zero-initialized residual adapters.
- There is no layer averaging, token compression, slot pooling, shared projection, or free-generating answer head.
- Writer training combines frozen Native functional distillation, a temporary span/support probe, token relation and K/V binding preservation, variance floor, and projection regularization.
- The formal checkpoint contains only the Writer. The temporary probe and optimizer are discarded.
- Canonical memories are recached from the frozen Writer, then a fresh independently initialized probe is trained.

## Grid

`all36` and `uniform16` use identical training budgets at seeds 1234, 2345, and 3456. Each fresh probe reports correct, question-only, zero, shuffled, K/V mismatch, synchronized token permutation, and layer permutation. Retention uses the corresponding P3-B Native-minus-zero F1 denominator.

Cross-sample shuffled source-answer hit is reported but is not a hard validity gate: an unrelated current question does not identify which fact was the answer to the source sample's question.

```bash
bash /home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234/run_all.sh all
```

```bash
bash /home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234/run_all.sh status
```
