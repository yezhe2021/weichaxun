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
| `test_sanity.py` | 审查§五 | 最小单元测试：方向元数据 / checkpoint 往返 / Writer 梯度 / Self Identity |

## 关键机制（复用已验证的 8B→4B 实现）

- **pre-RoPE K**：`NativeCapture` 在 `self_attn` 前向用 `k_norm` 提取（`k_norm` 之后、RoPE 之前）；V 取自 `v_proj`。
- **注入**：Writer 输出 pre-RoPE K + V 恢复 Receiver 尺度后，用 Receiver 的 `rotary_emb` 重新旋转为 post-RoPE K，构造 `DynamicCache`。
- **协议**：`apply_chat_template(enable_thinking=False)`；`context_input_ids`（Sender 输入，无 Question）与 `question_suffix_ids`（Receiver prompt）由 offset mapping 精确定界。

## 运行（服务器，固定解释器）

```bash
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
cd /home/yezhe/不同模型的kv分析/scripts

# 0) 单元测试（正式实验前必跑）
$PY -u test_sanity.py                                  # CPU：方向元数据 / checkpoint / 梯度
$PY -u test_sanity.py --gpu --model <Qwen3-0.6B路径>    # GPU：Self Identity 一致

# 冒烟：验证 Direct CE → Stage A → Stage B → Final evaluation 全链路（smoke=16/4/4，gate 不强制通过）
$PY -u run_pipeline.py --experiment 06_17 --mode smoke --stage all

# 正式：分步执行（formal 仅在对应方向 gate 通过后运行）
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage init
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage audit
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage cache
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage baselines
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage gates      # 四方向全部记录
$PY -u run_pipeline.py --experiment 06_17 --mode development --stage formal --direction 06_to_17
```

## Smoke 模式说明（审查 §四/§五）

- Smoke 规模 16/4/4：train 满足 overfit 的 Bridge/Comparison 各 8 条；validation/test 每类 ≥2 条，donor derangement 可构造。
- Smoke 阶段 Gate 不强制通过（只验证链路：forward / 梯度 / optimizer / checkpoint 保存加载 / validation 生成 / evaluation 输出）。
- Development Gate 失败：方向记入 `gate.json`，不中断其它方向；Formal 仅对通过方向继续，Gate 缺失直接报错。

## 训练与门控要点（审查修复）

- 训练路径 Writer 前向不置于 `no_grad` 下；CE/表示损失校验 `requires_grad`，首步校验参数梯度非零。
- 过拟合 eval 固定在训练集，不参与 validation checkpoint 选择；Gate 用训练集 best checkpoint，只诊断，正式训练重新 Identity 初始化。
- Self 方向用独立 Gate（Update0 与 Native 一致率 100%、Update0 KV max error ≈ 0、训练后 F1 不降超 2 点、Correct > Shuffled）；跨模型方向用 TrainRecovery / Specificity / NLL ratio。
- Stage6 按方向显式指定 Sender/Receiver（`17_to_06` 的 Sender 是 1.7B）；EM 用 Normalized EM。

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
