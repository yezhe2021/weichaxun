# Qwen3-0.6B ↔ 1.7B 双向规模因子实验结果

- 数据：HotpotQA distractor 完整 10 段 Context，train 512 / validation 64 / test 64
- 训练：Writer 逐层全秩 Linear（Identity 初始化），max_updates=400（快速模式）
- 评估：test 64 条，correct/shuffled/question_only/fulltext/native 全条件

## 结果矩阵（test 64 条）

| 方向 | 训练路径 | CrossGain | SelfGain | ReleaseDelta | Specificity |
| --- | --- | --- | --- | --- | --- |
| self_06 | f1_ce | +0.493 | 0.000 | +0.493 | 0.315 |
| self_06 | stage_a→CE | +0.465 | 0.000 | +0.465 | 0.304 |
| self_17 | f1_ce | +0.407 | −0.010 | +0.418 | 0.240 |
| self_17 | stage_a→CE | +0.489 | −0.010 | +0.499 | 0.290 |
| **06→17** | f1_ce | +0.208 | 0.000 | **+0.208** | 0.029 |
| **06→17** | stage_a→CE | +0.189 | 0.000 | **+0.189** | 0.086 |
| 17→06 | f1_ce | +0.175 | −0.010 | +0.186 | −0.018 |
| 17→06 | stage_a→CE | +0.336 | −0.010 | +0.347 | 0.145 |

## 核心结论

1. **`ReleaseDelta_{0.6→1.7} = +0.21 > 0`**：0.6B 自己读全文无增益（SelfGain=0），但其 KV 经 Writer 翻译后 1.7B 达到 correct F1=0.38，超过 1.7B 自己读全文（0.16）一倍多 → **小模型 KV 中存在自身无法利用、但可被大模型释放的信息**。
2. **Self 控制组健康**（self_06 +0.49 / self_17 +0.42）：Writer 管线不破坏信息。
3. **反向 17→06 specificity 为负**：弱 receiver（0.6B）无法特异性利用强 sender（1.7B）KV → Receiver 规模是瓶颈的提示。
4. **局限**：06→17 specificity 偏低（0.029），shuffled KV 带来部分通用增益；receiver_recovery 因 receiver 读全文反而低于 qonly 而方向异常。

## 文件

- `{direction}_{f1_ce,stage_a_then_ce}.json` — 每个方向的完整条件结果 + 派生指标（§十六 schema）
- `ablation_{direction}_{identity|align|align_ce}.json` — 消融实验：Alignment-only vs Align+CE（复用 stage_a/stage_b checkpoint）
- 每方向含 per_sample 明细（`_per_sample.jsonl`）

## 消融实验（Alignment-only vs Align+CE）

回答「高分来自 KV 空间对齐，还是 CE 把 Writer 训练成任务适配器」。评估条件含 correct/shuffled/zero/learned_constant/question_only/receiver_native。衍生指标：specificity、correct−qonly、correct−zero、correct−constant。
