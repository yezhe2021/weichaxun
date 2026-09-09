import argparse
from pathlib import Path

from p3d3_common import read_json, write_json
from p3e_n3_common import VALID_MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    modes = {
        mode: read_json(root / "evaluation" / mode / "SUCCESS.json")
        for mode in VALID_MODES
    }
    write_json(
        root / "SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N3 Attention-vs-Readout Sufficiency Ablation",
            "reader_c_frozen": True,
            "native_kv_excluded": True,
            "same_cache_samples_loss_epochs_seed": True,
            "tasks": ["support_tokens", "answer_span", "yes_no"],
            "modes": modes,
        },
    )


if __name__ == "__main__":
    main()

