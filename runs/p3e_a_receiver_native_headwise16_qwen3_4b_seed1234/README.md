# P3-E-A Receiver Native Headwise16

Stage A validates the uncompressed native GQA external-attention path before any Head-Structured Writer is introduced.

```text
Qwen3-4B Evidence-only prefill
  -> Native pre-RoPE K / native V [16,T,8,128]
  -> Qwen3-4B pre-RoPE Query [B,S,32,128]
  -> per-Query-head rank-32 residual adapter (still 128d)
  -> fixed GQA mapping: each four Query heads read one KV head
  -> readout [B,S,32,128]
  -> frozen layer-native o_proj [4096,2560]
  -> direct scalar gate
  -> extra self-attention branch before DecoderLayer residual
```

Only Query adapters and 16 scalar gates are trainable. There is no Canonical projection, output adapter, layer router, compatibility gate, or residual MLP.

The run first caches 512 train and 64 independent validation samples, performs a 16-sample overfit gate, and starts the formal 512-sample training only if the native path passes that gate.

```bash
cd /home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
nohup bash run_all.sh all > p3e_a_run.log 2>&1 & echo $! > p3e_a_run.pid
```

```bash
tail -f /home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234/p3e_a_run.log
```
