from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, read_jsonl, save_json


def aggregate_conditions(records):
    rows = [cond for record in records for cond in record.get("conditions", [])]
    conditions = {}
    for name in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == name]
        def mean(key, subset=None):
            vals = [x[key] for x in (subset or group)]
            return sum(vals) / len(vals) if vals else float("nan")
        conditions[name] = {
            "count": len(group),
            "em": mean("em"),
            "f1": mean("f1"),
            "nll": mean("nll"),
            "bridge_f1": mean("f1", [x for x in group if x["type"] == "bridge"]),
            "comparison_f1": mean("f1", [x for x in group if x["type"] == "comparison"]),
        }
    return conditions


def aggregate_equivalence(records):
    clone = [r["clone_gate"] for r in records]
    cont = [r["continuity"] for r in records]
    matches = [r["generation"][-1]["exact_generation_match"] for r in records]

    def mean(rows, key):
        return sum(x[key] for x in rows) / len(rows) if rows else float("nan")

    return {
        "samples": len(records),
        "generation_exact_match_rate": sum(matches) / len(matches) if matches else float("nan"),
        "clone_gate": {
            "logits_max_abs": max(x["logits_max_absolute_error"] for x in clone),
            "cache_nmse": max(x["post_forward_B_equals_C_nmse"] for x in clone),
            "base_unmutated_max_abs": max(x["base_unmutated_max_abs"] for x in clone),
            "top1_match": mean(clone, "top1_match_rate"),
        },
        "continuity": {
            "mean_kl": mean(cont, "mean_kl"),
            "max_kl": max(x["max_kl"] for x in cont),
            "top1_match_rate": mean(cont, "top1_match_rate"),
            "nll_diff": max(x["nll_absolute_difference"] for x in cont),
            "logits_max_abs": max(x["logits_max_absolute_error"] for x in cont),
        },
    }


def recovery(conditions):
    base = conditions.get("question_only", {}).get("f1", float("nan"))
    denom = conditions.get("full_hybrid_replay", {}).get("f1", float("nan")) - base
    out = {}
    for name in ("fa_only", "fa_recurrent", "full_hybrid_replay", "delta_only"):
        if name not in conditions:
            continue
        num = conditions[name]["f1"] - base
        out[name] = num / denom if denom else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    root = Path(cfg["work_dir"])
    equivalence = read_jsonl(root / "outputs" / args.mode / "equivalence_samples.jsonl")
    ablations = read_jsonl(root / "outputs" / args.mode / "ablation_samples.jsonl")

    conditions = aggregate_conditions(ablations)
    equiv = aggregate_equivalence(equivalence)
    recovery_map = recovery(conditions)

    fq = conditions.get("question_only", {})
    fh = conditions.get("full_hybrid_replay", {})
    sh = conditions.get("shuffled", {})
    gaps = {
        "correct_minus_shuffled_em": fh.get("em", float("nan")) - sh.get("em", float("nan")),
        "correct_minus_shuffled_f1": fh.get("f1", float("nan")) - sh.get("f1", float("nan")),
        "correct_minus_question_only_em": fh.get("em", float("nan")) - fq.get("em", float("nan")),
        "correct_minus_question_only_f1": fh.get("f1", float("nan")) - fq.get("f1", float("nan")),
    }

    g = cfg["gates"]
    checks = {
        "clone_logits_max_abs": equiv["clone_gate"]["logits_max_abs"] <= g["clone_logits_max_abs"],
        "clone_cache_nmse": equiv["clone_gate"]["cache_nmse"] <= g["clone_cache_nmse"],
        "replay_mean_kl": equiv["continuity"]["mean_kl"] <= g["replay_mean_kl"],
        "answer_nll_diff": equiv["continuity"]["nll_diff"] <= g["answer_nll_diff"],
        "top1_match_rate": equiv["continuity"]["top1_match_rate"] >= g["top1_match_rate"],
        "generation_match": equiv["generation_exact_match_rate"] >= g["generation_match_min"],
        "em_diff": abs(fh.get("em", 0) - conditions.get("continuous", {}).get("em", 0)) <= g["em_f1_diff"],
        "f1_diff": abs(fh.get("f1", 0) - conditions.get("continuous", {}).get("f1", 0)) <= g["em_f1_diff"],
    }
    gate = {"passed": all(checks.values()), "checks": checks, "thresholds": g}

    summary = {
        "experiment": cfg["experiment_name"], "mode": args.mode,
        "conditions": conditions, "recovery": recovery_map, "gaps": gaps,
        "equivalence": equiv, "gate": gate,
    }
    save_json(root / "metrics" / f"{args.mode}_metrics.json", summary)

    print(f"\n[{args.mode}] conditions (EM | F1 | NLL):")
    for name in sorted(conditions):
        c = conditions[name]
        print(f"  {name:22s} {c['em']:.3f} | {c['f1']:.3f} | {c['nll']:.3f}")
    print(f"  {'recovery':22s} " + " | ".join(f"{k}={v:.3f}" for k, v in recovery_map.items()))
    print(f"  gaps: " + ", ".join(f"{k}={v:.3f}" for k, v in gaps.items()))
    print(f"equivalence: {equiv}")
    print(f"gate: passed={gate['passed']} failed={[k for k, v in checks.items() if not v]}")


if __name__ == "__main__":
    main()
