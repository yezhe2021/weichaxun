import argparse
import csv
import re
from pathlib import Path


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    return float(value) if value not in {"", "nan", "NaN", None} else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Plot free-running layerwise logit-lens diagnosis")
    parser.add_argument("--summary", default=str(Path(__file__).resolve().parent / "results" / "free_running_layerwise_summary.csv"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "plots"))
    parser.add_argument("--steps", default="0,1,2,4,8,16,32")
    parser.add_argument("--metrics", default="mean_gold_margin,mean_gold_prob,mean_gold_rank,prefix_survival_rate_before_step")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [int(item) for item in args.steps.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    methods = sorted({row["method"] for row in rows})

    for step in steps:
        selected = [row for row in rows if int(row["step"]) == step]
        if not selected:
            continue
        for metric in metrics:
            plt.figure(figsize=(8, 4.5))
            plotted = False
            for method in methods:
                method_rows = sorted([row for row in selected if row["method"] == method], key=lambda row: int(row["layer"]))
                if not method_rows:
                    continue
                xs = [int(row["layer"]) for row in method_rows]
                ys = [to_float(row[metric]) for row in method_rows]
                plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=2.5, label=method)
                plotted = True
            if not plotted:
                plt.close()
                continue
            plt.xlabel("receiver layer")
            plt.ylabel(metric)
            plt.title(f"free-running step {step}: {metric}")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(out / f"step_{step:03d}__{sanitize(metric)}.png", dpi=180)
            plt.close()


if __name__ == "__main__":
    main()
