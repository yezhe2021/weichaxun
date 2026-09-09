from dataclasses import dataclass

import torch

from units import build_synchronized_units, select_units


@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int
    special: bool = False


def main():
    text = "The player was playing football"
    llama = [Span(0, 0, 0, True), Span(1, 0, 3), Span(2, 3, 10), Span(3, 10, 14),
             Span(4, 14, 19), Span(5, 19, 22), Span(6, 22, 31)]
    qwen = [Span(0, 0, 0, True), Span(1, 0, 3), Span(2, 3, 10), Span(3, 10, 14),
            Span(4, 14, 22), Span(5, 22, 31)]
    units = build_synchronized_units(text, llama, qwen)
    playing = next(unit for unit in units if unit.raw_text == " playing")
    assert playing.llama_indices == (4, 5) and playing.qwen_indices == (4,)
    assert playing.llama_right == 5 and playing.qwen_right == 4
    unicode_units = build_synchronized_units("甲乙abc", [Span(0, 0, 0, True), Span(1, 0, 1), Span(2, 1, 2), Span(3, 2, 5)],
                                             [Span(0, 0, 0, True), Span(1, 0, 2), Span(2, 2, 5)])
    assert unicode_units[0].byte_end == len("甲乙".encode("utf-8"))
    scores = torch.arange(8).float()
    chosen = select_units(units, scores, "llama", "max", 2, True)
    assert len(chosen) == len(set(unit.unit_id for unit in chosen)) == 2
    padded = select_units(units[:2], scores, "llama", "max", 4, True)
    assert len(padded) == 4 and len(set(unit.unit_id for unit in padded)) == 2
    print("Synchronized unit byte-boundary/selection tests passed", flush=True)


if __name__ == "__main__": main()
