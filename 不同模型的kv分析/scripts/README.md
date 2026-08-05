# 跨模型 KV 规模因子实验 · 代码

双向、同结构、跨规模的因子实验：拆解 **Sender 写入的信息量 / Receiver 利用信息的能力 / Writer 翻译损耗**。

- 第一组：Qwen3-0.6B ↔ Qwen3-1.7B（28 层，KV 各 8×128=1024 维，逐层映射）
- 第二组：Qwen3-4B ↔ Qwen3-8B（36 层，KV 同 1024 维，逐层映射）
- 数据：HotpotQA distractor 完整 10 段 Context（开发 512/64/64）

## 文件

| 文件 | 阶段 | 作用 |
| --- | --- | --- |
| `protocol.py` | 全阶段 | 数据协议（chat template + System/User）、manifest、pre-RoPE KV 提取、KV 注入、生成、指标、Store |
| `writer.py` | §七 | 逐层全秩 Linear Writer（Identity 初始化，RMS scale 归一化/恢复） |
| `stage0_token_audit.py` | §四 | 跨模型 tokenizer/协议审计（`all_samples_identical=true` 才继续） |
| `stage1_baselines.py` | §五 | 模型自身能力基线 + Cache Gate（FullText≈Official≈Manual） |
| `stage2_cache_assets.py` | §六 | 生成 context-only 的 pre-RoPE K + V 资产（FP16）与 RMS scale |
| `train_writer.py` | §八/九/十 | 训练：overfit（16 样本功能 Gate）/ direct（CE）/ stage_a（表示对齐）/ stage_b（CE） |
| `stage6_evaluate.py` | §十一/十二 | test 全条件（correct/shuffled/no_memory/fulltext/native）+ 派生指标 |
| `run_pipeline.py` | §十五 | 单 V100 执行顺序编排（init→audit→cache→baselines→gates→formal） |

## 关键机制（复用已验证的 8B→4B 实现）

- **pre-RoPE K**：`NativeCapture` 在 `self_attn` 前向用 `k_norm` 提取（`k_norm` 之后、RoPE 之前）；V 取自 `v_proj`。
- **注入**：Writer 输出 pre-RoPE K + V 恢复 Receiver 尺度后，用 Receiver 的 `rotary_emb` 重新旋转为 post-RoPE K，构造 `DynamicCache`。
- **协议**：`apply_chat_template(enable_thinking=False)`；`context_input_ids`（Sender 输入，无 Question）与 `question_suffix_ids`（Receiver prompt）由 offset mapping 精确定界。

## 运行（服务器，固定解释器）

```bash
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
cd /home/yezhe/不同模型的kv分析/scripts

# 冒烟：验证全流程能跑通
$PY -u run_pipeline.py --experiment 06_17 --mode smoke --stage all

# 正式：分步执行（gate 通过才继续 formal）
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage init
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage audit
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage cache
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage baselines
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage gates
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage formal --direction 06_to_17
```

## 输出布局（§十六）

```
runs/qwen3_06b_17b_bidirectional_kv_scale_diagnosis_seed1234/
├── artifacts/
│   ├── manifest.json / protocol.json / token_protocol_audit.json
│   ├── {sender,receiver}_baseline_summary.json
│   └── {smoke,development}/{self_06,self_17,06_to_17,17_to_06}/
│       ├── scales.pt  {overfit,direct,stage_a,stage_b}/
└── cache/{smoke,development}/{source_06,source_17,target_06,target_17}/
    └── {train,validation,test}/{id}.pt
└── evaluation/{direction}_{f1_ce,stage_a_then_ce}.json   # §十六 结果 schema
```

## 阶段门控（§四/五/九）

- 阶段0：`all_samples_identical = true`
- 阶段1：`top1_match ≥ 99.5%` 且 `mean_logit_KL < 1e-3` 且自由生成一致 100%
- 阶段4（16 样本功能 Gate）：`TrainRecovery ≥ 0.8` 且 `F1_correct − F1_shuffled ≥ 20` 且 `NLL_correct < 0.5·NLL_update0`
