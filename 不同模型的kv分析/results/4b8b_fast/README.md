# Qwen3-4B ↔ 8B 双向 KV 规模因子实验（快速方案 direct）

- 数据：HotpotQA distractor 完整 10 段 Context，train 512 / validation 64 / test 64
- 模型：Qwen3-4B（sender/receiver）、Qwen3-8B（sender/receiver），FP16，36 层 / 8 KV heads / head_dim 128 / feature_dim 1024
- 训练：Writer 逐层全秩 Linear（结构 Identity 初始化），`--training-path f1_ce`，max_updates=200（快速模式，direct 单路径，无 stage_a 两阶段）
- 评估：test 64 条，correct/shuffled/question_only/receiver_full_text/receiver_native/sender_full_text/sender_question_only 全条件
- 运行脚本：`run_4b8b_fast.sh`（完整复现命令），日志：`4b8b_fast.log`

## 结果矩阵（test 64 条，fast direct / f1_ce / 200 updates）

| 方向 | correct_f1 | sender_self_gain | cross_gain | release_delta | receiver_recovery | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| self_04 | 0.6578 | +0.182 | +0.237 | +0.055 | 1.300 | 0.281 |
| self_08 | 0.6504 | +0.239 | +0.201 | −0.038 | 0.841 | 0.294 |
| 04→08 | 0.6395 | +0.182 | +0.190 | +0.008 | 0.796 | 0.282 |
| 08→04 | 0.6215 | +0.239 | +0.200 | −0.039 | 1.101 | 0.338 |

> 说明：sender_self_gain / cross_gain 在 self 方向上数值相同（sender=receiver）。release_delta 在 self_08 与 08→04 为负，源于 receiver_full_text 高于 qonly 基线，具体解读由结果分析自行判断。

## 文件

- `{direction}_fast_direct.json` — 每方向完整条件结果 + 派生指标（§十六 schema）
- `{direction}_fast_direct_per_sample.jsonl` — 每样本明细（correct/shuffled/qonly/fulltext 预测与 f1）
- `config.json` — 实验配置（路径、规模、模型结构）
- `run_4b8b_fast.sh` — 4 方向 direct 训练 + stage6 评估的完整运行命令
- `4b8b_fast.log` — 完整运行日志（含 4 方向训练验证曲线与评估输出）
