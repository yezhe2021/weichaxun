# P2-C1: Structurally Heterogeneous Qwen3-1.7B Writer

P2-C1 changes only the sender from the P2-B setup. The Qwen3-8B backbone and
the P2-A2 Query-only Reader are loaded from their existing checkpoint and kept
strictly frozen. Only a new Qwen3-1.7B sender-specific Writer is optimized.

## Fixed protocol

- Same balanced 512 train pairs, 64 test pairs, prompts, `FINAL:` prefix, seed,
  two epochs, and Qwen3-8B Native-KV teacher as P2-B.
- Sender sees Question + Evidence A + Evidence B.
- All evidence-token pre-RoPE K and native V are retained.
- No token, slot, layer, head, or communication-budget compression.

## Writer

For every Qwen3-8B Reader layer, the Writer selects five nearby sender layers
by normalized depth and learns independent K/V layer routing. It then applies
independent K/V head mixing, head-dimension projections, and per-layer per-head
scale calibration. Output tensors exactly match the frozen Reader interface.

The primary losses are frozen-receiver base/counterfactual generation,
answer-swap margins, and correct versus shuffled/mismatched memory ranking.
Native Qwen3-8B route, answer attention mass, readout, and token-aligned KV
losses are auxiliary only.

## Baseline and evaluation

`raw_minimal_1_7b` is a deterministic, parameter-free shape baseline using
normalized-depth nearest layers, proportional head repeat/selection, and
dimension truncation or zero padding. It is not a learned Writer.

Evaluation reports full text, Native 8B, Writer 1.7B, raw-minimal 1.7B,
shuffled, mismatched, zero, and Reader-off free-running generations, including
EM, paired consistency, EOS, original-answer leakage, wrong-memory answer hit,
attention mass, route KL, readout cosine, Native gap, recovery ratio, and a
paired bootstrap Writer-versus-raw interval.

Strong success requires Native-gap recovery >= 85%. Basic success requires
paired consistency >= 70% and a positive 95% bootstrap lower bound over raw.

## Scope

The output remains Qwen3-8B Reader-specific per-layer KV. This experiment does
not establish receiver-independent Canonical Evidence-KV.

Run:

```bash
cd /home/yezhe/伪查询
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python \
  bash runs/p2c1_structural_writer_qwen3_1_7b_to_8b_seed1234/run_all.sh all
```

Final result:

```text
runs/p2c1_structural_writer_qwen3_1_7b_to_8b_seed1234/eval_writer_qwen3_1_7b/SUCCESS.json
```
