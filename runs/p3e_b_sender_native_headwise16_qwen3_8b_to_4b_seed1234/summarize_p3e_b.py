import argparse

from p3d3_common import read_json, write_json


def extract(result):
    condition = result["conditions"]["correct_sender_native_headwise16"]
    return {"em": condition["em"], "f1": condition["f1"], "bridge_f1": condition["by_type"].get("bridge", {}).get("f1"),
            "comparison_f1": condition["by_type"].get("comparison", {}).get("f1"), "correct_shuffled_gap": result["correct_shuffled_f1_gap"],
            "question_only_gain": result["correct_question_only_f1_gain"], "prediction_switch_rate": result["prediction_switch_rate"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage-a", required=True); parser.add_argument("--zero-shot", required=True); parser.add_argument("--retrained", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); stage_a, zero, retrained = read_json(args.stage_a), read_json(args.zero_shot), read_json(args.retrained)
    stage_a_condition = stage_a["conditions"]["correct_receiver_native_headwise16"]
    result = {"status": "complete", "experiment": "P3-E Native Headwise Stage A/B comparison",
              "receiver_native_4b_to_4b": {"em": stage_a_condition["em"], "f1": stage_a_condition["f1"], "correct_shuffled_gap": stage_a["correct_shuffled_f1_gap"]},
              "sender_native_8b_to_4b_stage_a_reader_zero_shot": extract(zero), "sender_native_8b_to_4b_retrained_reader": extract(retrained)}
    result["diagnosis"] = "cross_model_directly_compatible" if extract(zero)["f1"] >= stage_a_condition["f1"] - 0.05 else ("reader_adaptation_recovers_cross_model_memory" if extract(retrained)["f1"] > extract(zero)["f1"] + 0.10 else "raw_cross_model_native_kv_incompatible_or_not_executable")
    write_json(args.out, result)


if __name__ == "__main__": main()
