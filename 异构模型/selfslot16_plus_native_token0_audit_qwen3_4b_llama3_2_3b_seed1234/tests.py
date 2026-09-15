import torch

from experiment import construct_condition


def main():
    native_key = torch.randn(2, 9, 3, 4)
    native_value = torch.randn_like(native_key)
    slot_key = torch.randn(2, 16, 3, 4)
    slot_value = torch.randn_like(slot_key)
    original = construct_condition("slot16_original", native_key, native_value, slot_key, slot_value)
    shifted = construct_condition("slot16_shifted_control", native_key, native_value, slot_key, slot_value)
    augmented = construct_condition("token0_plus_slot16", native_key, native_value, slot_key, slot_value)
    assert original[0].shape[1] == 16 and original[2].tolist() == list(range(16)) and original[3] == 16
    assert shifted[0].shape[1] == 16 and shifted[2].tolist() == list(range(1, 17)) and shifted[3] == 17
    assert augmented[0].shape[1] == 17 and augmented[2].tolist() == list(range(17)) and augmented[3] == 17
    torch.testing.assert_close(augmented[0][:, 0], native_key[:, 0])
    torch.testing.assert_close(augmented[1][:, 0], native_value[:, 0])
    torch.testing.assert_close(augmented[0][:, 1:], shifted[0])
    torch.testing.assert_close(augmented[1][:, 1:], shifted[1])
    print({"passed": True, "checks": ["original_protocol", "shifted_control",
          "token0_prepend", "identical_shifted_slots", "kv_synchronized"]})


if __name__ == "__main__":
    main()
