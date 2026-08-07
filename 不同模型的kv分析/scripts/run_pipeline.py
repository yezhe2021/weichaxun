"""单 V100 执行顺序编排（方案 §十五）。

阶段：
  init      初始化实验目录（§十六 布局）+ manifest + config.json
  audit     阶段0：Tokenizer 与协议审计（跨模型 token 一致性）
  cache     阶段2：KV 资产缓存 + 每个方向的 RMS scale
  baselines 阶段1：sender/receiver 各自 Cache Gate + 文本基线（F1^QOnly/FullText/Shuffled/SelfGain）
  gates     阶段4：全部方向的 16 样本功能 Gate（每个方向都跑，失败也记录）
  formal    阶段5+6：对通过 Gate 的方向跑 Direct CE 与 Stage A→B，再在 test 上评估
  all       依序执行以上全部，formal 覆盖全部方向（问题 10）

用法：
  python -u run_pipeline.py --experiment 06_17 --mode smoke --stage all      # 冒烟验证全流程
  python -u run_pipeline.py --experiment 06_17 --mode development --stage gates
  python -u run_pipeline.py --experiment 06_17 --mode development --stage formal --direction 06_to_17
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from protocol import load_json, prepare_manifest, progress, save_json

SCRIPTS = Path(__file__).resolve().parent
PYTHON = "/home/yezhe/data/miniconda3/envs/attnkv/bin/python"

HOTPOT_TRAIN = "/home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json"
HOTPOT_DEV = "/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json"

MODELS = {
    "0.6B": "/home/yezhe/all_models/models/Qwen/Qwen3-0.6B",
    "1.7B": "/home/yezhe/all_models/models/Qwen/Qwen3-1.7B",
    "4B": "/home/yezhe/all_models/models/Qwen/Qwen3-4B",
    "8B": "/home/yezhe/all_models/models/Qwen/Qwen3-8B",
}

# 每组模型同架构、KV 维度同为 8×128=1024，逐层可映射（方案 §一）。
# direction: (名字, source 目录, target 目录, Sender 模型, Receiver 模型)  —— 问题 5
EXPERIMENTS = {
    "06_17": {
        "work_dir_name": "qwen3_06b_17b_bidirectional_kv_scale_diagnosis_seed1234",
        "sender": "0.6B",
        "receiver": "1.7B",
        "num_layers": 28,
        "roles": [("source_06", "0.6B"), ("source_17", "1.7B"), ("target_06", "0.6B"), ("target_17", "1.7B")],
        "directions": [
            ("self_06", "source_06", "target_06", "0.6B", "0.6B"),
            ("self_17", "source_17", "target_17", "1.7B", "1.7B"),
            ("06_to_17", "source_06", "target_17", "0.6B", "1.7B"),
            ("17_to_06", "source_17", "target_06", "1.7B", "0.6B"),
        ],
        "default_direction": "06_to_17",
    },
    "4b_8b": {
        "work_dir_name": "qwen3_4b_8b_bidirectional_kv_scale_diagnosis_seed1234",
        "sender": "4B",
        "receiver": "8B",
        "num_layers": 36,
        "roles": [("source_04", "4B"), ("source_08", "8B"), ("target_04", "4B"), ("target_08", "8B")],
        "directions": [
            ("self_04", "source_04", "target_04", "4B", "4B"),
            ("self_08", "source_08", "target_08", "8B", "8B"),
            ("04_to_08", "source_04", "target_08", "4B", "8B"),
            ("08_to_04", "source_08", "target_04", "8B", "4B"),
        ],
        "default_direction": "04_to_08",
    },
}

# 方案 §四：开发 512/64/64（扩大 512/128/256）；Smoke 保证每个 split 每类 ≥2 条，
# 且 train 满足 overfit 的 Bridge/Comparison 各 8 条，donor derangement 可构造（问题 4）。
SIZES = {
    "smoke": {"train": 16, "validation": 4, "test": 4},
    "development": {"train": 512, "validation": 64, "test": 64},
}


def run(args):
    command = [PYTHON, "-u"] + list(args)
    progress("$ " + " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def cfg_for(exp, mode, base):
    return {
        "sender_path": MODELS[exp["sender"]],
        "receiver_path": MODELS[exp["receiver"]],
        "hotpot_train": HOTPOT_TRAIN,
        "hotpot_dev": HOTPOT_DEV,
        "work_dir": str(base / exp["work_dir_name"]),
        "sizes": SIZES[mode],
        "smoke_sizes": SIZES["smoke"],
        "seed": 1234,
        "max_answer_tokens": 32,
        "max_new_tokens": 32,
        "receiver_dtype": "float16",
        "num_layers": exp["num_layers"],
        "num_kv_heads": 8,
        "head_dim": 128,
        "feature_dim": 1024,
    }


def direction_of(exp, direction):
    match = [d for d in exp["directions"] if d[0] == direction]
    if not match:
        raise ValueError(f"unknown direction {direction}")
    return match[0]


def stage_init(cfg, exp, mode):
    work = Path(cfg["work_dir"])
    for directory in ("artifacts", "cache", "checkpoints", "evaluation"):
        (work / directory).mkdir(parents=True, exist_ok=True)
    save_json(work / "config.json", cfg)
    prepare_manifest(cfg)
    progress("init done: manifest + config frozen")


def stage_audit(cfg, exp, mode):
    work = Path(cfg["work_dir"])
    audit_out = work / "artifacts" / "token_protocol_audit.json"
    run([
        str(SCRIPTS / "stage0_token_audit.py"),
        "--sender-model", cfg["sender_path"],
        "--receiver-model", cfg["receiver_path"],
        "--data", cfg["hotpot_dev"],
        "--out", str(audit_out),
        "--max-samples", "64" if mode == "smoke" else "256",
    ])
    audit = load_json(audit_out)
    if not audit["all_samples_identical"]:
        raise RuntimeError(f"token protocol audit failed: {audit['mismatches']}")
    progress(f"token protocol audit passed: {audit['checks']}")


def stage_cache(cfg, exp, mode):
    """阶段2：缓存每个模型 KV 一次（同一模型 target 目录软链接到 source），再按方向统计 scale。"""
    work = Path(cfg["work_dir"])
    cached = {}
    for role_dir, model_key in exp["roles"]:
        dest = work / "cache" / mode / role_dir
        if model_key in cached:
            if not dest.exists():
                dest.symlink_to(cached[model_key], target_is_directory=True)
            continue
        run([
            str(SCRIPTS / "stage2_cache_assets.py"),
            "--workdir", cfg["work_dir"], "--mode", mode,
            "--source-dir", role_dir, "--target-dir", role_dir,
            "--model", MODELS[model_key],
            "--num-layers", str(cfg["num_layers"]),
            "--feature-dim", str(cfg["feature_dim"]),
            "--sampled-tokens", "128",
            "source",
        ])
        cached[model_key] = dest
    # 每个方向统计 RMS scale（source / target 组合不同）
    for direction, source_dir, target_dir, *_ in exp["directions"]:
        run([
            str(SCRIPTS / "stage2_cache_assets.py"),
            "--workdir", cfg["work_dir"], "--mode", mode,
            "--source-dir", source_dir, "--target-dir", target_dir,
            "--direction", direction,
            "--num-layers", str(cfg["num_layers"]),
            "--feature-dim", str(cfg["feature_dim"]),
            "--sampled-tokens", "128",
            "scales",
        ])
    progress("cache + scales done")


def stage_baselines(cfg, exp, mode):
    """阶段1：sender / receiver 各自 Cache Gate + 文本基线。"""
    work = Path(cfg["work_dir"])
    for role in ("sender", "receiver"):
        model_path = cfg[f"{role}_path"]
        run([
            str(SCRIPTS / "stage1_baselines.py"),
            "--model", model_path,
            "--workdir", cfg["work_dir"],
            "--split", "validation",
            "--out", str(work / "artifacts" / f"{role}_baseline"),
            "--max-samples", "0" if mode == "development" else "2",
            "--max-new-tokens", str(cfg["max_new_tokens"]),
        ])
        summary = load_json(work / "artifacts" / f"{role}_baseline_summary.json")
        progress(f"{role} baselines: {summary['f1']} self_gain={summary['self_gain']} cache_gate_passed={summary['cache_gate']['cache_gate_passed']}")
        if not summary["cache_gate"]["cache_gate_passed"]:
            if mode == "development":
                raise RuntimeError(f"{role} cache gate failed")
            progress(f"WARNING {role}: cache gate not passed in smoke (link check only)")


def _train(cfg, exp, mode, direction, phase, max_updates):
    _, source_dir, target_dir, _, receiver_model = direction_of(exp, direction)
    run([
        str(SCRIPTS / "train_writer.py"),
        "--workdir", cfg["work_dir"], "--mode", mode,
        "--phase", phase, "--direction", direction,
        "--receiver-model", MODELS[receiver_model],
        "--source-dir", source_dir, "--target-dir", target_dir,
        "--num-layers", str(cfg["num_layers"]),
        "--feature-dim", str(cfg["feature_dim"]),
        "--sampled-tokens", "128",
        "--max-updates", str(max_updates),
        "--max-new-tokens", str(cfg["max_new_tokens"]),
    ])


def stage_gates(cfg, exp, mode, max_updates):
    """阶段4：全部方向的 16 样本功能 Gate（问题 6：失败也继续，全部记录）。"""
    work = Path(cfg["work_dir"])
    for direction, *_ in exp["directions"]:
        _train(cfg, exp, mode, direction, "overfit", max_updates)
        gate = load_json(work / "artifacts" / mode / direction / "overfit" / "gate.json")
        progress(f"{direction} gate: recovery={gate.get('train_recovery')} specificity={gate.get('specificity')} passed={gate['passed']}")
    progress("gates done")


def stage_formal(cfg, exp, mode, direction, max_updates):
    """阶段5+6：Direct CE 与 Stage A→B，再在 test 上评估（方案 §十五 C0-C4）。

    Development 严格门控：gate 缺失或未通过则禁止继续（问题 6）。
    Smoke 只验证链路，不检查 gate。
    """
    work = Path(cfg["work_dir"])
    _, source_dir, target_dir, sender_model, receiver_model = direction_of(exp, direction)

    if mode == "development":
        gate_path = work / "artifacts" / mode / direction / "overfit" / "gate.json"
        if not gate_path.exists():
            raise RuntimeError(f"missing required gate: {gate_path}")
        gate = load_json(gate_path)
        if not gate["passed"]:
            progress(f"{direction} overfit gate NOT passed, skipping formal training")
            return

    # C1 路径 F1：Direct CE
    _train(cfg, exp, mode, direction, "direct", max_updates)
    # C2 路径 F2：Stage A（表示对齐）→ Stage B（CE）
    _train(cfg, exp, mode, direction, "stage_a", max_updates)
    _train(cfg, exp, mode, direction, "stage_b", max_updates)

    # C3/C4 阶段6 评估：同一 validation 选出的唯一 checkpoint，test 上跑全部条件。
    # Sender/Receiver 按 direction 显式指定（问题 5）。
    for training_path, phase_dir in (("f1_ce", "direct"), ("stage_a_then_ce", "stage_b")):
        checkpoint = work / "artifacts" / mode / direction / phase_dir / "best.pt"
        if not checkpoint.exists():
            progress(f"missing {checkpoint}, skip {training_path} evaluation")
            continue
        run([
            str(SCRIPTS / "stage6_evaluate.py"),
            "--workdir", cfg["work_dir"], "--mode", mode,
            "--receiver-model", MODELS[receiver_model],
            "--sender-model", MODELS[sender_model],
            "--writer-checkpoint", str(checkpoint),
            "--training-path", training_path,
            "--source-dir", source_dir, "--target-dir", target_dir,
            "--num-layers", str(cfg["num_layers"]),
            "--feature-dim", str(cfg["feature_dim"]),
            "--out", str(work / "evaluation" / f"{direction}_{training_path}.json"),
        ])
    progress(f"formal {direction} done")


def main():
    parser = argparse.ArgumentParser(description="单 V100 执行顺序编排")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--base", default="/home/yezhe/不同模型的kv分析/runs")
    parser.add_argument("--stage", choices=("init", "audit", "cache", "baselines", "gates", "formal", "all"), required=True)
    parser.add_argument("--direction")
    parser.add_argument("--max-updates", type=int, default=800, help="训练 updates（快速模式 400）")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    cfg = cfg_for(exp, args.mode, Path(args.base))
    direction = args.direction or exp["default_direction"]

    if args.stage in ("init", "all"):
        stage_init(cfg, exp, args.mode)
    if args.stage in ("audit", "all"):
        stage_audit(cfg, exp, args.mode)
    if args.stage in ("cache", "all"):
        stage_cache(cfg, exp, args.mode)
    if args.stage in ("baselines", "all"):
        stage_baselines(cfg, exp, args.mode)
    if args.stage in ("gates", "all"):
        stage_gates(cfg, exp, args.mode, args.max_updates)
    if args.stage == "all":
        # 问题 10：--stage all 遍历全部方向完成 2×2 矩阵
        for direction_name, *_ in exp["directions"]:
            stage_formal(cfg, exp, args.mode, direction_name, args.max_updates)
    elif args.stage == "formal":
        stage_formal(cfg, exp, args.mode, direction, args.max_updates)
    progress("pipeline complete")


if __name__ == "__main__":
    main()
