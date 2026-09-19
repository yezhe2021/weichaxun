import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--p3en", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    probe = read_json(args.probe)
    p3en = read_json(args.p3en)
    write_json(
        args.out,
        {
            "status": "complete",
            "experiment": "P3-E-N2 Reader Readout Bottleneck Diagnosis",
            "probe_evaluation": probe,
            "p3en_generation_reference": p3en,
            "decision_rule": {
                "readout_probe_high_generation_low": "Reader read information; residual/frozen Receiver cannot use it",
                "readout_probe_low": "Reader query-to-memory retrieval failed",
            },
            "automatic_conclusion_deferred": True,
        },
    )


if __name__ == "__main__":
    main()
