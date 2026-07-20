import argparse

from .data import prepare_dataset, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    rows = prepare_dataset(args.raw, args.out, args.limit, args.seed)
    removable = sum(row["removed_evidence_text"] is not None for row in rows)
    write_json(args.out + ".meta.json", {
        "source": args.raw, "samples": len(rows), "seed": args.seed,
        "answer_sentence_removal_coverage": removable / max(1, len(rows)),
    })


if __name__ == "__main__":
    main()
