import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3ep", required=True)
    parser.add_argument("--c2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p3ep = read_json(args.p3ep)
    c2 = read_json(args.c2)
    previous = p3ep["diagnostic_table"]
    table = {
        "reader_c_native_kv": previous["reader_c_native_kv"],
        "a1_av_trajectory": previous["a1_correct_kv"],
        "c1_pooled_128d_fullkv": previous["c1_correct_kv"],
        "c1_pooled_128d_hard_shuffled": previous["c1_hard_shuffled"],
        "c2_uncompressed_multislot": c2["conditions"]["c2_correct_kv"],
        "c2_uncompressed_multislot_hard_shuffled": c2["conditions"][
            "c2_hard_shuffled"
        ],
        "full_evidence_text": previous["full_evidence_text"],
    }
    c2_correct = table["c2_uncompressed_multislot"]
    c2_shuffled = table["c2_uncompressed_multislot_hard_shuffled"]
    c2_correct["correct_shuffled_f1_gap"] = (
        c2_correct["f1"] - c2_shuffled["f1"]
    )
    reader_f1 = table["reader_c_native_kv"]["f1"]
    text_f1 = table["full_evidence_text"]["f1"]
    c2_correct["automatic_f1_gap_recovery"] = (
        (c2_correct["f1"] - reader_f1) / (text_f1 - reader_f1)
        if abs(text_f1 - reader_f1) > 1e-12
        else None
    )
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-P-C2 Uncompressed Multi-Slot Trajectory Reader",
            "comparison_table": table,
            "state_and_slot_diagnostics": c2[
                "state_and_slot_diagnostics"
            ],
            "c1_preserved_as_pooled_memory_baseline": True,
            "c2_slots": 8,
            "c2_slot_dim": 256,
            "c2_slot_pooling_before_state_query": False,
            "p3ep_original_result": args.p3ep,
            "c2_evaluation": args.c2,
        },
    )


if __name__ == "__main__":
    main()

