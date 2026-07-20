import argparse
import json
from pathlib import Path

from hotpot_common import write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args()
    root = Path(args.root)
    files = {
        "q4_question_only": root / "evaluation/qwen3_4b_question_only/SUCCESS.json",
        "q4_full_text": root / "evaluation/qwen3_4b_full_text/SUCCESS.json",
        "q8_full_text": root / "evaluation/qwen3_8b_full_text/SUCCESS.json",
        "w8_to_r4_canonical": root / "evaluation/w8_to_r4_canonical/SUCCESS.json",
    }
    results = {}
    for name, path in files.items():
        with path.open(encoding="utf-8") as handle: results[name] = json.load(handle)
    write_json(root / "SUCCESS.json", {
        "status": "complete", "experiment": "P2-Hotpot W8 canonical Writer to frozen R4 Reader zero-shot",
        "weights_frozen": True, "samples": 64, "evidence_regime": "gold_supporting_sentences", "results": results,
    })


if __name__ == "__main__": main()
