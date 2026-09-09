# q35_full_hybrid_cache_replay_audit_seed1234

Qwen3.5-4B 完整 Hybrid Cache 提取—重注入—消融实验。

## 三个问题
1. Qwen3.5 完整 Context 执行后保存了哪些状态（FA KV / Delta recurrent / conv）
2. 这些状态能否独立提取并重新注入、等价恢复行为（Gate 1 cache 复制门 + Gate 2 任意切分重放门）
3. 各组件贡献多少（消融）

## 关键实现决策
- **切分**：单次 chat-template tokenize + offset mapping 找 `Question:` 边界（`render()`），
  保证 `full == prefix + suffix`。
- **重放**：`forward_logits` 里 Path A（连续，`use_cache=False`）与 Path B（prefill prefix → replay suffix，
  `use_cache=True`）。position_ids 由 `cache.get_seq_length()` 自动推导，**不手传 cache_position**。
- **clone**：`clone_cache()` 按 `layer_types` 重建 `DynamicCache`，逐字段复制
  FA 层 `keys/values` 与 linear 层 `conv_states/recurrent_states` + 初始化 flag。
- **置零**：`zero_components()` 同形状置零，每条件从新鲜 clone 出发（前向会原地更新 cache）。
- **数值环境**：`fla`/`causal-conv1d` 未安装 → torch fallback（chunk_size=64，state 在 fp32 计算）。
  非 64 对齐切分只有浮点归约顺序差异（~1e-7），无结构性相位偏移。

## 运行
```bash
bash run_with_cuda_resume.sh smoke    # Smoke-1 audit + Smoke-4 等价 + 消融
bash run_with_cuda_resume.sh formal   # 64 条正式
```
阶段 marker 在 `artifacts/stage_markers/`，CUDA 中断后重跑同一命令即续跑。

## 产物
- `cache_manifest.json`：每层 cache 结构（形状/dtype/norm/init flag）
- `metrics/smoke1_audit.json`：Gate 1 clone + Gate 2 split 诊断
- `outputs/{mode}/equivalence_samples.jsonl`：A/B/C logits + 生成一致性
- `outputs/{mode}/ablation_samples.jsonl`：各条件 EM/F1/NLL
- `metrics/{mode}_metrics.json`：汇总表 + recovery + gates

## Gate 值（config.json）
| 指标 | 门槛 |
| --- | --- |
| clone logits max abs | 1e-5 |
| clone cache nmse | 1e-6 |
| replay mean KL | 1e-4 |
| answer NLL 差 | 1e-3 |
| teacher-forced top-1 | 99.9% |
| greedy 生成一致率 | 98% |
| EM/F1 差 | 0.02 |
