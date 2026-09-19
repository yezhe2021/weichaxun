import argparse

from p3d3_common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--reader-c", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--p3em", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    p3em = read_json(args.p3em)
    evaluation = read_json(args.evaluation)
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-N Native Oracle Bottleneck Decomposition",
            "audit": read_json(args.audit),
            "reader_c_training": read_json(args.reader_c),
            "same_model_evaluation": evaluation,
            "heterogeneous_p3em_evaluation": p3em,
            "reader_c_checkpoint_sha256": file_sha256(
                read_json(args.reader_c)["checkpoint"]
            ),
            "decomposition": {
                "text_to_same_model_interface_gap_f1": (
                    evaluation["conditions"]["full_evidence_text"]["f1"]
                    - evaluation["conditions"][
                        "qwen3_4b_qconditioned_native_reader_c"
                    ]["f1"]
                ),
                "same_model_to_heterogeneous_gap_f1": (
                    evaluation["conditions"][
                        "qwen3_4b_qconditioned_native_reader_c"
                    ]["f1"]
                    - p3em["conditions"][
                        "question_conditioned_native_reader_b"
                    ]["f1"]
                ),
            },
            "n2_executed": False,
        },
    )


if __name__ == "__main__":
    main()
