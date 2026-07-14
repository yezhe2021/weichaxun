# P2-B: Heterogeneous Sender Writer

P2-B freezes the Qwen3-8B backbone and the best P2-A2 Query-only Reader. It
trains only a sender-specific Writer that converts complete Qwen3-4B pre-RoPE K
and native V into the per-layer KV geometry consumed by that fixed Reader.

## Writer

The Writer preserves every evidence token and emits one K/V memory per receiver
layer. K and V use independent trainable components:

- local normalized-depth layer routing;
- KV-head mixing;
- head-dimension projection;
- per-layer, per-head scale calibration.

When sender and receiver KV geometry matches, layer/head/dimension mappings start
near identity. No token, slot, head, or layer compression is performed.

## Training

The primary objective is answer generation through the frozen receiver and
Reader. It includes base/counterfactual NLL, answer-swap margins, and correct
versus shuffled/mismatched memory ranking. Frozen Qwen3-8B Native-KV provides
auxiliary query-conditioned route, external readout, and optional token-aligned
KV losses.

## Evaluation

The same P2-A2 test pairs are used for full text, Qwen3-8B Native-KV, Writer
Qwen3-4B KV, raw/shape-only Qwen3-4B KV, shuffled, mismatched, zero, and
Reader-off free-running conditions. Results include EM, paired counterfactual
consistency, EOS, per-sample generations, answer-token attention mass,
route/readout diagnostics, and Native-KV gap recovery.

## Scope

This experiment tests whether a new heterogeneous sender can join a frozen
Qwen3-8B Reader-compatible per-layer KV interface. The Writer output is not a
receiver-independent Canonical Evidence-KV, and P2-B does not establish unseen
Writer-Reader composition.

Run:

```bash
cd /home/yezhe/伪查询
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python \
  bash runs/p2b_heterogeneous_writer_qwen3_4b_to_8b_seed1234/run_all.sh all
```

Final result:

```text
runs/p2b_heterogeneous_writer_qwen3_4b_to_8b_seed1234/eval_writer_qwen3_4b/SUCCESS.json
```
