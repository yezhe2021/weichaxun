import torch

from train_p2c2_writer import key_mismatched_memory


wrong = {
    "keys": [torch.randn(2, 2, 4)],
    "values": [torch.randn(2, 2, 4)],
}
positive = {
    "keys": [torch.randn(2, 4, 4)],
    "values": [torch.randn(2, 4, 4)],
    "answer_token_mask": torch.tensor([False, False, True, False]),
}
result = key_mismatched_memory(wrong, positive)
assert result["keys"][0].shape == result["values"][0].shape == (2, 4, 4)
assert result["answer_token_mask"].tolist() == [False, False, True, False]
print("key_mismatch_test=passed")
