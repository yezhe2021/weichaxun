# P2-G1: Qwen3-4B Native Query Reader

This directory contains only the receiver-side Native Reader prerequisite.
The heterogeneous Qwen3-8B to Qwen3-4B Writer is a separate P2-G2 experiment.

## Setup

- Frozen evidence encoder and receiver: Qwen3-4B.
- Trainable module: rank-32 layer-wise Query adapter and scalar gates.
- Memory: all evidence-token pre-RoPE K and native V from all 36 layers.
- Data: 512 train pairs and 64 test pairs from the fixed P2-A2 split.
- Evaluation: full text, correct/CF KV, shuffled, mismatched, zero, and Reader-off.
- Result: paired consistency 0.890625; the 0.90 gate failed by one pair.

## Layout

```text
cache/   Qwen3-4B Native-KV cache
train/   Query-only Reader training outputs
eval/    free-running evaluation and gate result
```

## Run

```bash
cd /home/yezhe/伪查询
ROOT=runs/p2g1_native_reader_qwen3_4b_seed1234
chmod +x "$ROOT/run_all.sh"
nohup bash "$ROOT/run_all.sh" wait-cuda > "$ROOT/logs/run.log" 2>&1 &
echo $! > "$ROOT/run.pid"
```

The downstream P2-G2 experiment references this directory but is not stored in it.
