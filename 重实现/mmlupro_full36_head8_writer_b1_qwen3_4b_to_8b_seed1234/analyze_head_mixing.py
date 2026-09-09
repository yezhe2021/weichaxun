from __future__ import annotations

import argparse
from pathlib import Path
import torch
from common import cuda, load_json, save_json
from writers import load_writer, make_writer


def summary(rows, family):
    values = [row[family] for row in rows]
    return {
        "mean_cross_over_total": sum(row["cross_over_total"] for row in values) / len(values),
        "mean_mixing_entropy": sum(row["mixing_entropy"] for row in values) / len(values),
        "same_head_remains_largest_rate": sum(row["largest_source_head"] == index % 8 for index, row in enumerate(values)) / len(values),
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = load_json(args.config)
    paths = {"stage_a": cfg["stage_a_checkpoint"], "b1_final": str(Path(cfg["work_dir"]) / "checkpoints/quick/full36_head8/stage_b_final/best.pt")}
    result = {}
    for name, path in paths.items():
        writer = make_writer("full36_head8", cfg).to(cuda()).eval(); load_writer(path, writer)
        diagnostic = writer.head_mixing_diagnostics()
        result[name] = {"checkpoint": path, "summary": {family: summary(diagnostic["rows"], family) for family in ("k", "v")}, **diagnostic}
        del writer; torch.cuda.empty_cache()
    save_json(Path(cfg["work_dir"]) / "artifacts/head_mixing_diagnostics.json", result)


if __name__ == "__main__": main()
