import argparse

from p3d3_common import file_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--reader-a", required=True)
    parser.add_argument("--reader-b", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-M Fresh Native Reader Q-Conditioning Diagnosis",
            "audit": read_json(args.audit),
            "reader_a_training": read_json(args.reader_a),
            "reader_b_training": read_json(args.reader_b),
            "evaluation": read_json(args.evaluation),
            "reader_a_checkpoint_sha256": file_sha256(
                read_json(args.reader_a)["checkpoint"]
            ),
            "reader_b_checkpoint_sha256": file_sha256(
                read_json(args.reader_b)["checkpoint"]
            ),
            "core_variable": "whether frozen Qwen3-8B Sender sees Question",
            "transmitted_question_kv": False,
            "writer_or_canonical_used": False,
        },
    )


if __name__ == "__main__":
    main()
