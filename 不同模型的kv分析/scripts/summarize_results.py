"""汇总 evaluation/*.json 为完整结果矩阵（方案 §十三 归因）。

用法：
  python -u summarize_results.py --eval-dir <work_dir>/evaluation [--experiment 06_17]

输出：完整 4 方向 × 训练路径的派生指标表 + 按类型（Bridge/Comparison）分解。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 方向顺序（06_17：self_06, self_17, 06_to_17, 17_to_06）
DIRECTION_ORDER = ["self_06", "self_17", "06_to_17", "17_to_06"]
PATH_ORDER = ["f1_ce", "stage_a_then_ce"]


def parse_name(name):
    if name.endswith("_stage_a_then_ce"):
        return name[: -len("_stage_a_then_ce")], "stage_a_then_ce"
    if name.endswith("_f1_ce"):
        return name[: -len("_f1_ce")], "f1_ce"
    return name, name


def summarize(eval_dir):
    files = sorted(Path(eval_dir).glob("*.json"))
    files = [f for f in files if "_per_sample" not in f.name]
    rows = []
    for path in files:
        result = json.load(open(path))
        derived = result["derived"]
        by_type = result.get("by_type", {})
        direction, training_path = parse_name(path.stem)
        rows.append({
            "file": path.name,
            "direction": direction,
            "training_path": result.get("training_path", training_path),
            "n": result.get("n"),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in derived.items()},
            "bridge_release_delta": round(by_type.get("bridge", {}).get("release_delta", float("nan")), 4) if by_type else None,
            "comparison_release_delta": round(by_type.get("comparison", {}).get("release_delta", float("nan")), 4) if by_type else None,
            "bridge_specificity": round(by_type.get("bridge", {}).get("specificity", float("nan")), 4) if by_type else None,
            "comparison_specificity": round(by_type.get("comparison", {}).get("specificity", float("nan")), 4) if by_type else None,
        })
    return rows


def render(rows):
    lines = []
    header = ("direction".ljust(12) + "path".ljust(16) + "cross".rjust(7) + "self".rjust(7)
              + "relDelta".rjust(9) + "recovery".rjust(9) + "spec".rjust(7)
              + "br_delta".rjust(9) + "cmp_delta".rjust(10) + "br_spec".rjust(8) + "cmp_spec".rjust(8))
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float) and v != v:
                return "nan"
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(
            row["direction"].ljust(12) + str(row["training_path"]).ljust(16)
            + fmt(row["cross_gain"]).rjust(7) + fmt(row["sender_self_gain"]).rjust(7)
            + fmt(row["release_delta"]).rjust(9) + fmt(row["receiver_recovery"]).rjust(9)
            + fmt(row["specificity"]).rjust(7)
            + fmt(row["bridge_release_delta"]).rjust(9) + fmt(row["comparison_release_delta"]).rjust(10)
            + fmt(row["bridge_specificity"]).rjust(8) + fmt(row["comparison_specificity"]).rjust(8)
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()
    rows = summarize(args.eval_dir)
    rows.sort(key=lambda r: (DIRECTION_ORDER.index(r["direction"]) if r["direction"] in DIRECTION_ORDER else 99,
                             PATH_ORDER.index(r["training_path"]) if r["training_path"] in PATH_ORDER else 99))
    print(render(rows))


if __name__ == "__main__":
    main()
