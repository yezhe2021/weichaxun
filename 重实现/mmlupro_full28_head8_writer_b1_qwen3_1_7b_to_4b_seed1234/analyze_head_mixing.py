from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import cuda, load_json, save_json
from writers import load_writer, make_writer


def summarize(diagnostic):
    output = {}
    for family in ("k", "v"):
        rows = [row[family] for row in diagnostic["rows"]]
        output[family] = {
            "mean_cross_over_total": sum(row["cross_over_total"] for row in rows) / len(rows),
            "mean_mixing_entropy": sum(row["mixing_entropy"] for row in rows) / len(rows),
            "same_head_remains_largest_rate": sum(row["largest_source_head"] == index % 8 for index, row in enumerate(rows)) / len(rows),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    paths = {
        "stage_a": cfg["stage_a_checkpoint"],
        "b1_final": str(Path(cfg["work_dir"]) / "checkpoints/quick/full28_head8/stage_b_final/best.pt"),
    }
    result = {}
    for name, path in paths.items():
        writer = make_writer("full28_head8", cfg).to(cuda()).eval()
        load_writer(path, writer)
        diagnostic = writer.head_mixing_diagnostics()
        result[name] = {"checkpoint": path, "summary": summarize(diagnostic), **diagnostic}
        del writer
        torch.cuda.empty_cache()
    save_json(Path(cfg["work_dir"]) / "artifacts/head_mixing_diagnostics.json", result)


if __name__ == "__main__":
    main()
