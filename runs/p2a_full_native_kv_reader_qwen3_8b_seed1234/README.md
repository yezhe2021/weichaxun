# P2-A: Full Native-KV External Reader Upper Bound

This experiment tests whether a frozen Qwen3-8B receiver can use complete, uncompressed, same-model native evidence K/V through a per-layer external attention branch.

## Controlled variables

- Sender and receiver are the same frozen Qwen3-8B.
- Sender input is question + evidence A + evidence B, without the answer suffix.
- Every layer caches evidence-token pre-RoPE K and native V.
- Receiver sees only the question.
- External Q and K are both pre-RoPE.
- Native GQA broadcasting and frozen native `o_proj` are reused.
- The default smoke run trains only 36 scalar gates.
- No canonical space, compression, heterogeneous mapping, or multi-sender composition is used.

## Default smoke size

```text
train: 64 counterfactual pairs
test: 16 counterfactual pairs
epochs: 1
generation: 24 tokens
```

Run:

```bash
cd /home/yezhe/伪查询
bash runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/run_all.sh smoke
```

Inspect:

```bash
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/text_gate/SUCCESS.json
cat runs/p2a_full_native_kv_reader_qwen3_8b_seed1234/eval_gate_only/SUCCESS.json
```

The text gate should be checked before interpreting Reader training. The native-KV cache can be reused across later gate and low-rank Reader experiments.
