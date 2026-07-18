# P2-G2: Qwen3-8B to Qwen3-4B Reverse Native-KV Writer

This is an independent heterogeneous Writer experiment. It does not contain the
P2-G1 Native Reader experiment or duplicate its cache/results.

## Dependencies

- Frozen sender cache: P2-A2 Qwen3-8B full evidence-token Native KV.
- Frozen receiver: Qwen3-4B.
- Frozen Reader and 4B Native-KV teacher cache:
  `runs/p2g1_native_reader_qwen3_4b_seed1234`.
- Trainable parameters: 8B-to-4B Writer only.

P2-G1 measured 0.890625 paired consistency and missed its 0.90 gate by one pair.
Proceeding with that frozen Reader is an explicit experimental override recorded
in `AUDIT.json`.

## Variants

- `matched_task_only`: task losses dominate; Reader diagnostics are weak and sparse.
- `reader_aligned`: additionally aligns route, external readout, and target attention mass.

Both variants use the same 512 train pairs, 64 test pairs, seed, initialization,
steps, and learning rate. Qwen3-8B and Qwen3-4B evidence token IDs are required to
match exactly.

## Layout

```text
teacher_stats/  4B Native K calibration statistics
train/          Writer checkpoints and lightweight histories
eval/           free-running generations and Reader diagnostics
comparison/     unified comparison of both variants
```

## Run

```bash
cd /home/yezhe/伪查询
ROOT=runs/p2g2_reverse_writer_qwen3_8b_to_4b_seed1234
chmod +x "$ROOT/run_all.sh"
nohup bash "$ROOT/run_all.sh" wait-cuda > "$ROOT/logs/run.log" 2>&1 &
echo $! > "$ROOT/run.pid"
```

Writer outputs remain specific to the frozen Qwen3-4B Reader. This experiment
does not establish receiver-independent Canonical Evidence-KV.
