import argparse
import csv
import re
from pathlib import Path


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "NaN"} else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Plot layerwise logit-lens curves")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", default="mean_gold_margin,mean_gold_prob,mean_gold_rank")
    parser.add_argument("--token-groups", default="critical_tokens,first_answer_token")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "PLOT_SKIPPED.txt", "w", encoding="utf-8") as handle:
            handle.write(f"matplotlib unavailable: {exc}\n")
        return

    rows = read_rows(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    token_groups = [item.strip() for item in args.token_groups.split(",") if item.strip()]
    datasets = sorted({row["dataset"] for row in rows})
    modes = sorted({row["receiver_prompt_mode"] for row in rows})

    for dataset in datasets:
        for mode in modes:
            for token_group in token_groups:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["receiver_prompt_mode"] == mode
                    and row["token_group"] == token_group
                ]
                if not selected:
                    continue
                methods = sorted({row["method"] for row in selected})
                for metric in metrics:
                    plt.figure(figsize=(8, 4.5))
                    plotted = False
                    for method in methods:
                        method_rows = sorted(
                            [row for row in selected if row["method"] == method],
                            key=lambda row: int(row["layer"]),
                        )
                        xs = [int(row["layer"]) for row in method_rows]
                        ys = [f(row, metric) for row in method_rows]
                        if xs:
                            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=2.5, label=method)
                            plotted = True
                    if not plotted:
                        plt.close()
                        continue
                    plt.xlabel("receiver layer")
                    plt.ylabel(metric)
                    plt.title(f"{dataset} {mode} {token_group} {metric}")
                    plt.grid(True, alpha=0.25)
                    plt.legend(fontsize=7)
                    plt.tight_layout()
                    name = f"{sanitize(dataset)}__{sanitize(mode)}__{sanitize(token_group)}__{sanitize(metric)}.png"
                    plt.savefig(out / name, dpi=180)
                    plt.close()


if __name__ == "__main__":
    main()
