import argparse
import json
from pathlib import Path

from p2iw_common import write_json


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = {
        "random_projection": root / "baselines/random/SUCCESS.json",
        "pca_projection": root / "baselines/pca/SUCCESS.json",
        "hidden_teacher": root / "baselines/teacher/SUCCESS.json",
        "small_overfit": root / "small_overfit/SUCCESS.json",
        "full_writer_fresh_probe": root / "fresh_probe/SUCCESS.json",
        "diagnostics": root / "diagnostics/SUCCESS.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing P2-I-W results: {missing}")
    results = {name: load(path) for name, path in paths.items()}
    fresh = results["full_writer_fresh_probe"]
    geometry = results["diagnostics"]["splits"]["test"]
    decision = {
        "fresh_probe_metric_gate": fresh["fresh_probe_success"],
        "noncollapse_gate": not geometry["collapsed_by_p2i_threshold"],
    }
    decision["p2iw_passed"] = bool(all(decision.values()))
    write_json(root / "SUCCESS.json", {"status": "complete", "decision": decision, "results": results})


if __name__ == "__main__":
    main()
