from __future__ import annotations

import argparse
from pathlib import Path

from common import append_jsonl, fixed_samples, load_json, load_tokenizer, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    tok = load_tokenizer(cfg)
    samples = fixed_samples(cfg, tok)
    out_path = Path(cfg["work_dir"]) / "artifacts" / "rendered_samples.jsonl"
    for sample in samples:
        append_jsonl(out_path, sample)
    progress(f"rendered {len(samples)} fixed samples -> {out_path}")


if __name__ == "__main__":
    main()
