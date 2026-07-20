# Hidden-State-to-Public-KV MVP

This package implements an independent, token-preserving Public-KV channel. It does not translate or modify native model KV caches.

## Fixed geometry

- A0 sender/receiver: Qwen3-4B.
- A1 sender: Qwen3.5-4B; receiver: Qwen3-4B.
- Qwen3 taps/readers: `3,8,12,17,21,26,30,35` (zero-based block indices).
- Qwen3.5 taps: `3,7,11,15,19,23,27,31`.
- Public KV: 8 heads x 128 dimensions at every tap.
- Public Q: 32 heads x 128 dimensions; 4 query groups per KV head.
- All backbone parameters are frozen. No token pooling, routing, native K/V reuse, or Public RoPE is used.

## Data

```bash
python -m hidden_public_kv.prepare_data \
  --raw /home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json \
  --out runs/hidden_public_kv_mvp/data/train512.jsonl --limit 512

python -m hidden_public_kv.prepare_data \
  --raw /home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json \
  --out runs/hidden_public_kv_mvp/data/dev500.jsonl --limit 500 --seed 1235
```

The sender input is exactly the ordered gold supporting sentence text. Titles, question text, distractor paragraphs, and answers are not added to sender input.

## A0

Start with a 32-example overfit run:

```bash
python -m hidden_public_kv.train_a0 \
  --model /home/yezhe/all_models/models/Qwen/Qwen3-4B \
  --data runs/hidden_public_kv_mvp/data/train512.jsonl \
  --out runs/hidden_public_kv_mvp/a0_overfit --limit 32 --epochs 20 \
  --gradient-accumulation 8 --optimizer adafactor --device cuda
```

## A1 hidden cache

```bash
python -m hidden_public_kv.cache_hidden \
  --kind qwen35 --model /home/yezhe/all_models/models/Qwen/Qwen3___5-4B \
  --data runs/hidden_public_kv_mvp/data/train512.jsonl \
  --out runs/hidden_public_kv_mvp/cache/qwen35_train --include-removed

python -m hidden_public_kv.train_a1 \
  --receiver-model /home/yezhe/all_models/models/Qwen/Qwen3-4B \
  --cache runs/hidden_public_kv_mvp/cache/qwen35_train/index.json \
  --a0-checkpoint runs/hidden_public_kv_mvp/a0/checkpoint_latest.pt \
  --out runs/hidden_public_kv_mvp/a1 --optimizer adafactor --device cuda
```

## Free-running evaluation

Cache the matching dev sender states, then run:

```bash
python -m hidden_public_kv.evaluate \
  --receiver-model /home/yezhe/all_models/models/Qwen/Qwen3-4B \
  --cache runs/hidden_public_kv_mvp/cache/qwen35_dev/index.json \
  --checkpoint runs/hidden_public_kv_mvp/a1/checkpoint_latest.pt \
  --out runs/hidden_public_kv_mvp/eval --max-new-tokens 32
```

The evaluator reports question-only, full gold text, correct Public KV, shuffled Public KV, zero Public KV, reader-off, and answer-sentence-removed conditions. The latter is reported only where normalized answer matching identifies a removable supporting sentence.

## 16 GB notes

- Use FP16, batch size one, Adafactor, gradient accumulation, and gradient checkpointing.
- A1 training uses cached detached sender states, so only the receiver resides on GPU.
- The ranking objective uses low-memory sequential backward passes rather than a `2B` receiver batch.
- Do not load Qwen3.5-4B and Qwen3-4B on the same 16 GB GPU.
