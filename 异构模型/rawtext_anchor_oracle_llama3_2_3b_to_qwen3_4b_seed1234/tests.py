import torch

from anchors import build_anchors, region_bounds, selected_mass
from offsets import TokenSpan, overlap, span_distance, span_iou


def run_tests():
    assert region_bounds(10, 4) == [(0, 2), (2, 5), (5, 7), (7, 10)]
    source = [TokenSpan(0, 99, 0, 0), TokenSpan(1, 1, 0, 3), TokenSpan(2, 2, 3, 5),
              TokenSpan(3, 3, 5, 8), TokenSpan(4, 4, 8, 10)]
    target = [TokenSpan(0, 8, 0, 2), TokenSpan(1, 9, 2, 4), TokenSpan(2, 10, 4, 7),
              TokenSpan(3, 11, 7, 9), TokenSpan(4, 12, 9, 10)]
    importance = torch.tensor([100., 1., 4., 3., 2.])
    anchors = build_anchors("abcdefghij", source, target, importance, 4)
    assert len(anchors) == 4
    assert all(anchor["source_index"] != 0 and anchor["target_index"] != 0 for anchor in anchors)
    assert anchors[0]["source_index"] == 1  # token0 is excluded despite its artificial high score
    assert anchors[0]["target_index"] == 1  # target token0 is also excluded; deterministic nearest fallback
    assert overlap(TokenSpan(0, 0, 1, 4), TokenSpan(0, 0, 3, 6)) == 1
    assert span_distance(2, TokenSpan(0, 0, 3, 6)) == 1
    assert span_iou(TokenSpan(0, 0, 1, 4), TokenSpan(0, 0, 3, 6)) == 1 / 5
    assert selected_mass(torch.tensor([9., 1., 2., 3.]), [1, 2, 2]) == 3 / 6
    return {"passed": True, "checks": ["balanced_raw_regions", "source_token0_exclusion",
            "target_token0_exclusion", "shared_character_anchor", "nearest_span_fallback",
            "span_overlap", "span_iou", "unique_attention_mass"]}


if __name__ == "__main__": print(run_tests())
