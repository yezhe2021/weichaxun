import argparse

from p3d3_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = read_json(args.zero_shot)
    metrics = result["metrics"]
    correct = metrics["correct_question_canonical"]["f1"]
    evidence = metrics["evidence_only_canonical"]["f1"]
    neutral = metrics["neutral_prefix_canonical"]["f1"]
    wrong = metrics["wrong_question_canonical"]["f1"]
    train = not (
        correct >= evidence + 0.05
        and correct >= neutral + 0.03
        and correct >= wrong + 0.03
    )
    write_json(
        args.out,
        {
            "status": "complete",
            "train_conditioned_writer": train,
            "automatic_rule": "skip only if correct_q exceeds evidence by .05 and neutral/wrong by .03 F1",
        },
    )


if __name__ == "__main__":
    main()
