# P2-I: Cached Canonical Evidence-KV Interface

This experiment replaces pairwise receiver-shaped KV Writers with one Qwen3-8B
sender Writer and two receiver-specific Readers.

## Interface

The Writer consumes all 36 x 8 Evidence-token pre-RoPE K/native V tensors from
Qwen3-8B and emits exactly:

```text
K_E: [256, 256]
V_E: [256, 256]
```

There is no receiver layer axis, attention-head axis, native head dimension,
RoPE coordinate, or tokenizer position in the serialized interface. Shared slot
assignment produces a state `z_m`; independent projections produce each slot's
address `k_m` and content `v_m`. Slot queries affect pooling weights but are not
added directly to slot values.

Qwen3-4B reads the slots at all 36 decoder layers. Qwen3.5-4B reads them only at
its eight real full-attention layers: `3,7,11,15,19,23,27,31`. Each Reader uses
shared query/output projections plus rank-32 per-layer residual adapters and a
scalar gate.

## Protocol

1. Audit model geometry, data hashes, cache alignment, and exact slot-permutation
   invariance.
2. Cache receiver-local functional teachers. Qwen3-4B uses its successful Native
   Query Reader. Qwen3.5-4B has no successful Native Reader checkpoint, so its
   first experiment uses full-text answer logits/states only and does not invent
   a route/readout teacher.
3. Train a mother model by alternating one Qwen3-4B epoch and one Qwen3.5-4B
   epoch. Only the active frozen backbone is loaded, which fits a 32 GB V100.
4. Freeze the final Writer, hash it, and cache every train/test Evidence-KV once.
5. Reinitialize and train each Reader independently from the same frozen slot
   cache. The training script asserts that the Writer hash and parameters do not
   change.
6. Train per-receiver joint Writer+Reader oracles from the mother checkpoint.
   These are capacity diagnostics, not the main method.
7. Run free-running controls and combine results. The four main evaluations must
   contain one identical Writer SHA-256.

The 512 original training pairs are not expanded: 448 are used for optimization
and 64 are reserved for validation/future cached screening. The original 64-pair
test set remains untouched. Default training uses live receiver forwards because
queries after earlier injections are state-dependent; cached question states and
teacher tensors are warm-start/diagnostic signals, not substitutes for final
free-running evaluation.

## Controls

Each receiver reports full text, correct slots, counterfactual slots, shuffled
complete memory, K/V-mismatched slots, a separately prefilled true A/B mismatch,
zero slots, Reader-off, joint slot permutation, and deletion of half the slots.
It saves every generation and per-layer Reader diagnostics. Slot permutation is
also checked directly on first-step logits; the expected maximum difference is
below `1e-5`.

## Run

```bash
cd /home/yezhe/伪查询
RUN=runs/p2i_cached_canonical_evidence_kv_qwen3_8b_seed1234
bash "$RUN/run_all.sh" audit
bash "$RUN/run_all.sh" all
```

For a detached run:

```bash
cd /home/yezhe/伪查询
RUN=runs/p2i_cached_canonical_evidence_kv_qwen3_8b_seed1234
mkdir -p "$RUN/logs"
nohup bash "$RUN/run_all.sh" wait-cuda >"$RUN/logs/p2i_run.log" 2>&1 &
echo $! >"$RUN/run.pid"
```

Status only:

```bash
cd /home/yezhe/伪查询
bash runs/p2i_cached_canonical_evidence_kv_qwen3_8b_seed1234/run_all.sh status
```

The primary success criterion is the frozen-Writer Reader retraining branch:
Qwen3-4B paired consistency at least 85%, Qwen3.5-4B at least 70%, near-zero
Reader-off/zero performance, correct base/CF switching, and a small oracle gain.
Success supports an initial `O(N_s + N_r)` interface claim for these two
receivers only; it does not yet establish multi-sender composition,
question-independent memory, or compressed `M=32` operation.
