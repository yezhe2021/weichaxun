import torch

from experiment import condition_indices, stride_indices


def main():
    assert stride_indices(6, 0).tolist() == [0, 2, 4]
    assert stride_indices(6, 1).tolist() == [1, 3, 5]
    assert stride_indices(5, 0).tolist() == [0, 2, 4]
    assert stride_indices(5, 1).tolist() == [1, 3]
    assert condition_indices(7, "keep_even").tolist() == [0, 2, 4, 6]
    assert condition_indices(7, "keep_odd").tolist() == [1, 3, 5]
    assert condition_indices(7, "keep_odd_plus_token0").tolist() == [0, 1, 3, 5]
    assert condition_indices(7, "keep_even_minus_token0").tolist() == [2, 4, 6]
    key = torch.randn(2, 7, 3, 4)
    value = torch.randn_like(key)
    indices = stride_indices(7, 0)
    assert key[:, indices].shape == value[:, indices].shape == (2, 4, 3, 4)
    assert torch.equal(key[:, indices][:, 1], key[:, 2])
    print({"passed": True, "checks": ["even_indices", "odd_indices", "odd_plus_token0",
          "even_minus_token0", "odd_length", "kv_synchronized"]})


if __name__ == "__main__":
    main()
