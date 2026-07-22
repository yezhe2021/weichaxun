# P3-E-B Sender Native Headwise16: Qwen3-8B to Qwen3-4B

Stage B keeps the Stage A native 128-dimensional GQA path unchanged and replaces only the Evidence encoder:

```text
Qwen3-8B Evidence-only Native KV [16,T,8,128]
  -> Qwen3-4B Native Query [B,S,32,128]
  -> per-Query-head rank-32 residual adapter (still 128d)
  -> fixed 4:1 GQA attention
  -> [B,S,4096]
  -> frozen Qwen3-4B layer-native o_proj
  -> scalar gate
```

No Writer, Canonical projection, head mixing, dimension compression, V adapter, layer router, compatibility gate, or residual MLP is used.

The existing Qwen3-8B cache is exposed through a lossless `[16,T,1024] -> [16,T,8,128]` view, avoiding duplicate multi-gigabyte caches.

The experiment reports:

1. Stage A Reader zero-shot on Qwen3-8B Native KV.
2. A freshly initialized Reader trained on Qwen3-8B Native KV, first on 16 samples and then on 512 samples if the gate passes.

```bash
cd /home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
nohup bash run_all.sh all > p3e_b_run.log 2>&1 & echo $! > p3e_b_run.pid
```

```bash
tail -f /home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234/p3e_b_run.log
```
