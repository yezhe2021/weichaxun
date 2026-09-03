# Receiver Cross-Attention Reader

Minimal heterogeneous-memory experiment with a frozen Qwen3-1.7B Sender and a frozen Qwen3-4B Receiver.

## Interface

- Sender input: evidence only.
- External memory: final normalized Sender hidden states, `[B,Tm,2048]`.
- Reader layers: zero-based Decoder layers `12,20,28,34`.
- Reader attention: 8 heads, 64 dimensions per head, shared K/V projections.
- Receiver LoRA layers: `12,13,20,21,28,29,34,35`.
- Receiver LoRA targets: native `q_proj` and `o_proj`, rank 8, alpha 16.
- Reader insertion: after the native self-attention residual and before the native MLP normalization.

The Receiver backbone and Sender are always frozen. The experiment trains three independent variants:

1. `reader_only`
2. `reader_lora`
3. `lora_only`

The joint checkpoint is also evaluated with the Reader disabled, and Reader conditions are evaluated with correct and strict hard-shuffled memory.

## Smoke test

```bash
cd /home/yezhe/伪查询/lora通信
bash run_all.sh smoke
```

The smoke test prepares 32 train and 32 validation examples, caches real Qwen3-1.7B memory, performs one optimizer step for every variant, and evaluates one validation sample under all conditions.
For tiny smoke splits only, the negative builder may fall back to a different answer type when no strict same-type candidate exists. Formal runs do not enable this fallback.

## Formal run

```bash
cd /home/yezhe/伪查询/lora通信
nohup bash run_all.sh all > outputs/formal.log 2>&1 &
```

Defaults are 512 training samples, 64 validation samples, five epochs, seed 1234, and CUDA. Stages can be run independently with `prepare`, `cache`, `negatives`, `train`, and `eval`.
