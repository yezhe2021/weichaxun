import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    zero = read_json(args.zero_shot)
    final = read_json(args.final)
    decision = read_json(args.decision)
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-L Question-Conditioned Sender Diagnosis",
            "zero_shot": zero,
            "final": final,
            "writer_training_decision": decision,
            "primary_manual_metric_pending": "C/P/W strict and lenient semantic accuracy",
            "single_core_variable": "Sender sees Question before Evidence KV extraction",
        },
    )


if __name__ == "__main__":
    main()
