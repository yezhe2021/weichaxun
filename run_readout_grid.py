import argparse
import json
import subprocess
from pathlib import Path


def csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default="/home/yezhe/data/miniconda3/bin/python")
    p.add_argument("--script", default="shared_latent_readout.py")
    p.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl")
    p.add_argument("--out-root", default="runs/grid_hotpotqa")
    p.add_argument("--layers", default="4,8,12,16")
    p.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    p.add_argument("--query-sources", default="no_context,oracle_full")
    p.add_argument("--memory-modes", default="topk,full")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--reader-hidden", type=int, default=256)
    p.add_argument("--cpu", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    layers = csv_ints(args.layers)
    alphas = csv_floats(args.alphas)
    query_sources = [x.strip() for x in args.query_sources.split(",") if x.strip()]
    memory_modes = [x.strip() for x in args.memory_modes.split(",") if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for query_source in query_sources:
        for memory_mode in memory_modes:
            for layer in layers:
                for alpha in alphas:
                    out = out_root / f"q-{query_source}_mem-{memory_mode}_l{layer}_a{alpha:g}"
                    cmd = [
                        args.python,
                        args.script,
                        "--data", args.data,
                        "--layer", str(layer),
                        "--alpha", str(alpha),
                        "--memory-mode", memory_mode,
                        "--query-source", query_source,
                        "--topk", str(args.topk),
                        "--max-length", str(args.max_length),
                        "--max-samples", str(args.max_samples),
                        "--eval-samples", str(args.eval_samples),
                        "--epochs", str(args.epochs),
                        "--reader-hidden", str(args.reader_hidden),
                        "--out", str(out),
                    ]
                    if args.cpu:
                        cmd.append("--cpu")
                    jobs.append({"out": str(out), "cmd": cmd})

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"prepared {len(jobs)} jobs; manifest={manifest}")
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {' '.join(job['cmd'])}")
        if not args.dry_run:
            subprocess.run(job["cmd"], check=True)


if __name__ == "__main__":
    main()
