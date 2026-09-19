from __future__ import annotations

import argparse
from pathlib import Path

from v2_common import (
    Stores, load_json, load_writer, rows_for, save_json, validate_rep
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    rows = rows_for(cfg, args.mode)
    writer = load_writer(cfg, args.mode, "v0")
    metrics = validate_rep(
        cfg, writer, Stores(cfg, args.mode, rows), rows["validation"], "a2"
    )
    save_json(
        Path(cfg["work_dir"]) / "artifacts" / args.mode / "v0"
        / "per_layer_metrics.json", metrics
    )


if __name__ == "__main__":
    main()
