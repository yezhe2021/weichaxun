# P3-E-K Canonical Information Sufficiency Probe

P3-E-K is a diagnostic probe, not a Reader and not a proposed communication
method. It asks whether the evidence required by HotpotQA is present in the
fixed Sender Native KV and learned C2 Canonical KV.

## Existing cache fields

The existing caches already contain:

- Canonical K/V `[16,T,16,128]`;
- Native K/V `[16,T,8,128]`;
- exact Evidence token IDs and character offsets;
- valid-token and official supporting-fact masks;
- original Evidence text and HotpotQA annotations.

They do not explicitly contain `sentence_id`. P3-E-K deterministically rebuilds
it from each `TITLE:` and numbered `[n]` sentence plus the stored token offsets.
No Native or Canonical K/V is regenerated.

The sidecar cache additionally stores frozen Qwen3-4B:

- the final hidden state of the last Question token `[2560]`;
- Evidence-only final hidden states `[T,2560]` for the full-text upper bound;
- sentence IDs and all token-aligned answer spans.

## Structured KV front-end

Canonical K/V preserves all 16 depth groups, 16 heads, and Evidence tokens.
For each `(layer, token, head)`, K and V are separately RMS-normalized,
concatenated, projected to 16 dimensions, and augmented with learned layer/head
identities. A frozen-Qwen Question vector conditions every structural unit by
elementwise multiplication. The fixed-order concatenation produces 4096
dimensions per token, followed by a 4096-to-512 MLP.

There is no layer or head average, sum, attention pooling, or learned routing.
Native 8-head KV is losslessly duplicated into adjacent 16-head structural
positions so it uses the same structured front-end.

All conditions share:

- a question-conditioned classification token;
- a 4-layer bidirectional 512-dimensional Token Transformer;
- supporting-token, answer-start, answer-end, and yes/no heads;
- identical seed, epochs, optimizer, and train/validation splits.

The full-text condition necessarily uses a modality-specific 2560-to-512 input
projection, but shares the complete Transformer and output-head architecture.

## Training and evaluation

- smoke16: 20 epochs per input mode;
- formal512: 5 epochs per input mode;
- validation: the fixed independent 64 samples;
- loss: `L_support + L_span + L_yesno`;
- span supervision marginalizes over all matching answer spans.

Conditions:

- `full_text_representation`
- `sender_native_kv`
- `learned_canonical_kv`
- `question_only_zero_memory`
- `canonical_sample_shuffled`
- `canonical_hard_shuffled`

Reported metrics include support-token F1/AUPRC, supporting-sentence Recall@2
and F1, span EM/token-F1/Top-5 hit, yes/no accuracy, current-answer/source-answer
tracking, and correct-versus-shuffled gaps.
