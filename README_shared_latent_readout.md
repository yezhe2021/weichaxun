# Single-Layer Shared Latent Memory Readout

Remote path conventions:

- Workspace and model root: `/home/yezhe/伪查询`
- Dataset root: `/home/yezhe/数据集`

This experiment freezes a sender model and a receiver model. It builds a compact shared memory `Z` from sender layer activations over `x + q`, then trains a lightweight receiver-side reader to reconstruct the receiver full-prefill attention output at one layer for the query tokens.

Default model paths:

- Sender: `/home/yezhe/伪查询/Qwen3-0.6B`
- Receiver: `/home/yezhe/伪查询/Qwen3-1.7B`

Example:

```bash
cd /home/yezhe/伪查询

/home/yezhe/data/miniconda3/bin/python shared_latent_readout.py \
  --sender-model /home/yezhe/伪查询/Qwen3-0.6B \
  --receiver-model /home/yezhe/伪查询/Qwen3-1.7B \
  --data /home/yezhe/数据集/swift/OpenHermes-2___5/openhermes2_5.json \
  --layer 12 \
  --max-samples 256 \
  --topk 128 \
  --epochs 1 \
  --out runs/slm_qwen3_06b_to_17b_l12
```

The remote machine currently has a usable dependency environment at:

```bash
/home/yezhe/data/miniconda3/bin/python
```

`python3 -m venv` is not usable on the system Python because `ensurepip` is missing. CUDA is also not visible in the current session (`torch.cuda.is_available() == False`, `nvidia-smi` fails with NVML initialization), so formal runs should wait until GPU visibility is fixed.

The workspace model paths are symlinks:

- `/home/yezhe/伪查询/Qwen3-0.6B` -> `/home/yezhe/C2C/baseline/Qwen3-0.6B`
- `/home/yezhe/伪查询/Qwen3-1.7B` -> `/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B`

The script reports:

- receiver full-prefill teacher CE
- receiver no-context CE
- shared `Z` + reader CE
- attention-output MSE/cosine
- final logit KL
- top-1 match
- approximate CUDA peak memory

Notes:

- The script is intentionally single-layer and keeps both LMs frozen.
- `Z` contains sender K/V projections, token position, saliency, and pooled sender hidden state.
- The reader does not restore receiver KV. It uses receiver query-token hidden states as queries and cross-attends into `Z`, producing a patch for the selected receiver layer attention output.
- The hook patches the receiver self-attention module output at `layer` only for query-token positions.
- OpenHermes conversation rows usually have no separate context `x`; for a real context-removal test, use rows with `context/query/answer` style fields or another dataset where `x` and `q` are distinct. Otherwise full-prefill and no-context can collapse to the same prompt.
