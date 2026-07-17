import argparse
import csv
import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def probe_row(path, branch, stage, representation):
    result = load(path)
    return {
        "branch": branch,
        "stage": stage,
        "representation": representation,
        "base_em": result["base_em"],
        "counterfactual_em": result["counterfactual_em"],
        "paired_consistency": result["paired_consistency"],
        "prediction_switch_rate": result["prediction_switch_rate"],
        "test_loss": result.get("test_loss"),
        "metric": result.get("weight_policy", "stage_specific_trained"),
    }


def generation_row(path, branch):
    result = load(path)
    conditions = {row["condition"]: row for row in result["conditions"]}
    return {
        "branch": branch,
        "stage": "free_running",
        "representation": "free_running_generation",
        "base_em": conditions["writer_base"]["accuracy"],
        "counterfactual_em": conditions["writer_counterfactual"]["accuracy"],
        "paired_consistency": result["paired_consistency"]["writer"],
        "prediction_switch_rate": None,
        "test_loss": None,
        "metric": "free_running_em",
    }


def first_large_drop(rows, threshold):
    for left, right in zip(rows, rows[1:]):
        drop = left["paired_consistency"] - right["paired_consistency"]
        if drop >= threshold:
            if left["stage"] == "raw_kv" and right["stage"] == "writer_kv":
                diagnosis = "Writer transformation is the first clear functional drop"
            elif left["stage"] == "writer_kv" and right["stage"] == "reader_readout":
                diagnosis = "Frozen Reader grounding is the first clear functional drop"
            elif left["stage"] == "reader_readout":
                diagnosis = "Injection or downstream generation is the first clear functional drop"
            elif left["stage"] == "sender_final_hidden" and right["stage"] == "raw_kv":
                diagnosis = "Lightweight raw-KV probe cannot reconstruct the final-hidden computation"
            else:
                diagnosis = f"First clear drop occurs at {left['stage']} -> {right['stage']}"
            return {
                "from": left["representation"],
                "to": right["representation"],
                "paired_drop": drop,
                "diagnosis": diagnosis,
            }
    return {
        "from": None,
        "to": None,
        "paired_drop": None,
        "diagnosis": "No >= threshold drop; quick audit is inconclusive",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-root", required=True)
    parser.add_argument("--task-chain-root", required=True)
    parser.add_argument("--shared-chain-root", required=True)
    parser.add_argument("--task-generation", required=True)
    parser.add_argument("--shared-generation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--drop-threshold", type=float, default=0.20)
    args = parser.parse_args()
    memory = Path(args.memory_root)
    common = [
        probe_row(
            memory / "sender_final_hidden" / "SUCCESS.json",
            "common", "sender_final_hidden", "llama_final_evidence_hidden",
        ),
        probe_row(
            memory / "raw_last_kv" / "SUCCESS.json",
            "common", "raw_kv_aux", "raw_llama_last_layer_kv",
        ),
        probe_row(
            memory / "raw_all_kv" / "SUCCESS.json",
            "common", "raw_kv", "raw_llama_all_28_layer_kv",
        ),
    ]
    branches = {}
    for name, writer_stage, chain_root, generation in (
        (
            "task_only", "writer_task_only_kv", Path(args.task_chain_root),
            Path(args.task_generation),
        ),
        (
            "shared_span_relation", "writer_shared_span_relation_kv",
            Path(args.shared_chain_root), Path(args.shared_generation),
        ),
    ):
        rows = [common[0], common[2]]
        rows.extend(
            [
                probe_row(
                    memory / writer_stage / "SUCCESS.json",
                    name, "writer_kv", f"{name}_writer_all_36_layer_kv",
                ),
                probe_row(
                    chain_root / "reader_all_readout" / "SUCCESS.json",
                    name, "reader_readout", f"{name}_reader_all_layer_readout",
                ),
                probe_row(
                    chain_root / "final_cumulative_delta" / "SUCCESS.json",
                    name, "cumulative_delta", f"{name}_final_cumulative_delta",
                ),
                probe_row(
                    chain_root / "receiver_final_hidden" / "SUCCESS.json",
                    name, "receiver_final_hidden", f"{name}_receiver_final_hidden",
                ),
                probe_row(
                    chain_root / "first_token_logits" / "SUCCESS.json",
                    name, "first_token_logits", f"{name}_first_token_logits",
                ),
                generation_row(generation / "SUCCESS.json", name),
            ]
        )
        branches[name] = {
            "rows": rows,
            "first_large_drop": first_large_drop(rows, args.drop_threshold),
        }
    table = common + [row for value in branches.values() for row in value["rows"][2:]]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "unified_stage_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with open(output / "first_large_drop.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": args.drop_threshold,
                "quick_audit_only": False,
                "branches": {
                    key: value["first_large_drop"] for key, value in branches.items()
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(output / "SUCCESS.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "experiment": "P2-E-C1 calibrated segmented functional readability audit",
                "table": table,
                "branches": branches,
                "claim_boundary": (
                    "The final-hidden positive control reuses the frozen Experiment-A "
                    "checkpoint. Incompatible representations use independently trained, "
                    "head-preserving or state-matched probes on the full 448/64/64 split."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
