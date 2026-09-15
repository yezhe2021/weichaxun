import torch

from ablation import build_condition
from modules import HybridChunkCompressor


def run_tests():
    module = HybridChunkCompressor(2, slots=4, heads=2, dim=4)
    key, value = torch.randn(2, 14, 2, 4), torch.randn(2, 14, 2, 4)
    selected = torch.tensor([1, 5, 7, 10])
    compact = build_condition(module, key, value, selected, "compact_native_only")
    matched = build_condition(module, key, value, selected, "position_matched_native_only")
    hybrid = build_condition(module, key, value, selected, "real_slot_hybrid")
    assert compact[0].shape[1] == 5 and compact[4] == 5 and compact[2].all()
    assert matched[0].shape[1] == 9 and matched[4] == 9
    assert matched[2].tolist() == [True, True, False, True, False, True, False, True, False]
    assert torch.count_nonzero(matched[0][:, 2::2]) == 0
    assert torch.count_nonzero(matched[1][:, 2::2]) == 0
    assert hybrid[0].shape[1] == matched[0].shape[1] and hybrid[4] == matched[4]
    torch.testing.assert_close(hybrid[0][:, 1::2], matched[0][:, 1::2])
    torch.testing.assert_close(hybrid[1][:, 1::2], matched[1][:, 1::2])
    return {"passed": True, "checks": ["compact33", "position_matched65", "true_zero_slots",
            "masked_slots", "identical_native_entries", "identical_suffix_start_C_D"]}


if __name__ == "__main__": print(run_tests())
