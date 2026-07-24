# P3-E-J Canonical Soft-Prefix Executability Diagnosis

This experiment is a diagnostic, not a proposed replacement communication
interface. It tests whether the fixed C2 question-independent Canonical memory
contains enough token-level evidence to reconstruct a Receiver-native prefix,
and whether such a prefix is executable by a completely frozen Qwen3-4B.

## Fixed components

- Qwen3-8B Sender and the C2 learned Writer are not loaded for training.
- Their existing Canonical caches `[16,T,16,128]` are read without modification.
- The C2 Writer checkpoint hash is recorded in every formal result.
- Qwen3-4B and the historical C1 Headwise Reader remain frozen.
- Canonical and Native caches are paired by sample ID and token length. Native
  metadata supplies the exact Evidence token IDs used when the caches were made.

## Trainable component

Only `SoftPrefixDecoder` is optimized:

1. separate K/V RMS normalization;
2. shared per-head K/V fusion;
3. learned pooling across 16 Canonical heads;
4. content-dependent mixing across 16 Canonical depth groups;
5. a two-layer bidirectional 512-dimensional Transformer token mixer;
6. projection to the Qwen3-4B embedding dimension and fixed RMS calibration.

The complete Evidence token axis is preserved. There is no slot compression,
question-aware selection, Sender update, Writer update, or Receiver update.

## Training

- Smoke: 16 samples, Stage A then Stage B, two epochs each.
- Formal Stage A: 512 samples, five epochs, embedding reconstruction only.
- Formal Stage B: initialize from Stage A and train five epochs with answer NLL,
  embedding preservation, Question-position hidden-state alignment at Decoder
  layers 8/16/24/35, and answer-position temperature-2 logit KL.
- The fixed validation64 split is used only after training.

Every epoch is checkpointed. `checkpoint_best.pt` is selected by training loss,
not validation performance.

## Evaluation conditions

- `question_only`
- `full_evidence_text`
- `exact_evidence_embeddings`
- `current_c1_headwise_reader`
- `canonical_soft_prefix`
- `sample_shuffled_canonical_soft_prefix`
- `token_order_shuffled_soft_prefix`
- `soft_prefix_off`

The preflight audit requires the real `input_ids` path and the exact
`inputs_embeds` path to match at prefill logits and the next cached decode step.
Validation additionally requires their complete greedy generations to match,
and requires `soft_prefix_off` to equal `question_only` token for token.

## Interpretation scope

The result separates decodability from executability:

- good embedding reconstruction and near-text QA implicates the historical
  external-residual interface;
- good reconstruction but C1-level QA shows that numerical input similarity
  alone does not restore a compatible reasoning trajectory;
- poor reconstruction shows that C2 Canonical memory does not preserve complete
  Evidence content.

No outcome is presented as evidence that KV-to-soft-token conversion should be
the final communication method.
