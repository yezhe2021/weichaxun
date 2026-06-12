import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    rows = []
    for path in Path(args.root).glob("**/metrics.json"):
        summary = json.load(open(path, encoding="utf-8")).get("summary", {})
        if "patched_ce" not in summary:
            continue
        rows.append({
            "path": str(path),
            "patched_ce": summary.get("patched_ce"),
            "patched_kl": summary.get("patched_kl"),
            "attn_mse": summary.get("attn_mse"),
            "teacher_ce": summary.get("teacher_ce"),
            "no_context_ce": summary.get("no_context_ce"),
            "layer": summary.get("layer"),
            "alpha": summary.get("alpha"),
            "query_source": summary.get("query_source"),
            "memory_mode": summary.get("memory_mode"),
            "memory_tokens": summary.get("memory_tokens"),
        })

    for row in sorted(rows, key=lambda x: x["patched_ce"])[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
