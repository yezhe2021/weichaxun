# P2-G: Reverse Qwen3-8B to Qwen3-4B Native-KV Communication

## Step 1

`step1_native_reader/` trains a rank-32 Query-only Reader for a frozen Qwen3-4B
receiver using Qwen3-4B Native KV. Its measured paired consistency is 0.890625.
The original 0.90 gate failed by one test pair; step 2 proceeds by explicit user
decision and records this override in `step2_8b_to_4b_writer/AUDIT.json`.

## Step 2

- Sender memory: cached full evidence-token Native KV from frozen Qwen3-8B.
- Receiver: frozen Qwen3-4B.
- Reader: frozen step-1 rank-32 Query-only Reader.
- Trainable parameters: 8B-to-4B Writer only.
- Data: the same 512 train pairs and 64 test pairs used by P2-A2/P2-G1.
- Token transport: strict identity; Qwen3-8B and Qwen3-4B evidence token IDs
  must match for every sample used during training and evaluation.
- Variants: `matched_task_only` and `reader_aligned`, initialized with the same seed.

The evaluation reports full text, 4B Native KV, raw/minimal 8B KV, Writer KV,
shuffled, K-mismatched, V-mismatched, zero, and Reader-off generation conditions,
plus paired consistency, prediction switching, EOS, memory-answer hits, route KL,
readout cosine, target attention mass, and Native-gap recovery.

Run in the background:

```bash
cd /home/yezhe/伪查询
ROOT=runs/p2g_reverse_qwen3_8b_to_4b_seed1234
chmod +x "$ROOT/run_step2.sh"
nohup bash "$ROOT/run_step2.sh" wait-cuda > "$ROOT/logs/step2.log" 2>&1 &
echo $! > "$ROOT/step2.pid"
```

Monitor:

```bash
bash runs/p2g_reverse_qwen3_8b_to_4b_seed1234/run_step2.sh status
tail -f runs/p2g_reverse_qwen3_8b_to_4b_seed1234/logs/step2.log
```

Writer outputs remain specific to the frozen Qwen3-4B Reader interface. This
experiment does not establish receiver-independent Canonical Evidence-KV.
