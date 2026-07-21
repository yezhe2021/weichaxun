import argparse
import json
import sys
from pathlib import Path


def read(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def overfit(args):
    report = read(args.train_success)
    passed = bool(report.get("overfit_pass"))
    write(args.out, {
        "status": "pass" if passed else "fail",
        "gate": "16-sample free-running overfit",
        "thresholds": report.get("overfit_thresholds"),
        "source": args.train_success,
    })
    return 0 if passed else 2


def final(args):
    oracle = read(args.oracle)
    native = read(args.native) if args.native else None
    candidates = []
    for item in args.candidate:
        name, path = item.split("=", 1)
        result = read(path)
        correct = result["correct"]
        gap = result["correct_minus_shuffled_f1"]
        bridge = correct.get("bridge", {}).get("f1", 0.0)
        native_f1 = None
        if native:
            native_f1 = native.get("conditions", {}).get("correct", {}).get("f1")
        checks = {
            "correct_f1_at_least_0_38": correct["f1"] >= 0.38,
            "correct_shuffled_gap_at_least_0_10": gap >= 0.10,
            "bridge_f1_at_least_0_25": bridge >= 0.25,
            "canonical_near_native": native_f1 is None or correct["f1"] >= native_f1 - 0.03,
        }
        candidates.append({
            "name": name,
            "source": path,
            "correct_f1": correct["f1"],
            "bridge_f1": bridge,
            "correct_shuffled_gap": gap,
            "native_f1": native_f1,
            "checks": checks,
            "minimum_success": all(checks.values()),
        })
    standard = oracle["modes"]["standard"]
    oracle_mode = oracle["modes"]["oracle_token_layer"]
    oracle_bridge_gain = oracle["oracle_gain"]["bridge_f1"]
    oracle_gap_gain = oracle["oracle_gain"]["correct_shuffled_gap_gain"]
    oracle_decisive = oracle_bridge_gain >= 0.05 or oracle_gap_gain >= 0.05
    successful = [item for item in candidates if item["minimum_success"]]
    if successful:
        disposition = "question-independent route meets the P3-D2 minimum standard"
    elif oracle_decisive:
        disposition = "stop after repair round 2; redesign the external Reader interface before more question-independent training"
    else:
        disposition = "stop question-independent Reader work; next optional experiment is q-aware Sender with a nontrivial dual-Sender split"
    report = {
        "status": "complete",
        "oracle_grounding_decisive": oracle_decisive,
        "oracle_bridge_f1_gain": oracle_bridge_gain,
        "oracle_correct_shuffled_gap_gain": oracle_gap_gain,
        "ordinary_oracle_f1_gap": oracle_mode["correct"]["f1"] - standard["correct"]["f1"],
        "candidates": candidates,
        "successful_candidates": [item["name"] for item in successful],
        "disposition_after_step3": disposition,
        "step4_executed": False,
    }
    write(args.out, report)
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    overfit_parser = subparsers.add_parser("overfit")
    overfit_parser.add_argument("--train-success", required=True)
    overfit_parser.add_argument("--out", required=True)
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--oracle", required=True)
    final_parser.add_argument("--native")
    final_parser.add_argument("--candidate", action="append", default=[])
    final_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    code = overfit(args) if args.command == "overfit" else final(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
