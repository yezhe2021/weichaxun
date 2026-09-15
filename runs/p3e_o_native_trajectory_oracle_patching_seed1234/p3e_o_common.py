from contextlib import contextmanager

import torch


def first_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_first_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def replace_last_token(output, state):
    tensor = first_tensor(output)
    patched = tensor.clone()
    patched[:, -1, :] = state.to(device=tensor.device, dtype=tensor.dtype)
    return replace_first_tensor(output, patched)


class NativeTrajectoryOracle:
    def __init__(self, model, selected_layers):
        self.model = model
        self.selected_layers = list(selected_layers)

    @contextmanager
    def capture(self, destination):
        handles = []
        for layer_index in self.selected_layers:
            layer = self.model.model.layers[layer_index]

            def attention_hook(module, args, kwargs, output, layer_index=layer_index):
                destination.setdefault(layer_index, {})["attention_output"] = (
                    first_tensor(output)[:, -1, :].detach().clone()
                )

            def post_attention_pre_hook(
                module, args, kwargs, layer_index=layer_index
            ):
                destination.setdefault(layer_index, {})[
                    "post_attention_state"
                ] = args[0][:, -1, :].detach().clone()

            def block_hook(module, args, kwargs, output, layer_index=layer_index):
                destination.setdefault(layer_index, {})["block_output"] = (
                    first_tensor(output)[:, -1, :].detach().clone()
                )

            handles.append(
                layer.self_attn.register_forward_hook(
                    attention_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    post_attention_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                layer.register_forward_hook(block_hook, with_kwargs=True)
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def patch(self, oracle_states, mode):
        if mode not in ("attention_output", "post_attention_state", "block_output"):
            raise ValueError(f"Unknown oracle patch mode: {mode}")
        missing = [
            layer
            for layer in self.selected_layers
            if mode not in oracle_states.get(layer, {})
        ]
        if missing:
            raise RuntimeError(f"Oracle states missing layers: {missing}")
        handles = []
        target_inputs = {}
        for layer_index in self.selected_layers:
            layer = self.model.model.layers[layer_index]

            if mode == "attention_output":
                def attention_hook(
                    module, args, kwargs, output, layer_index=layer_index
                ):
                    return replace_last_token(
                        output, oracle_states[layer_index]["attention_output"]
                    )

                handles.append(
                    layer.self_attn.register_forward_hook(
                        attention_hook, with_kwargs=True
                    )
                )

            elif mode == "post_attention_state":
                def layer_pre_hook(
                    module, args, kwargs, layer_index=layer_index
                ):
                    target_inputs[layer_index] = args[0][:, -1, :].detach()

                def attention_hook(
                    module, args, kwargs, output, layer_index=layer_index
                ):
                    if layer_index not in target_inputs:
                        raise RuntimeError("Target layer input was not captured")
                    required_attention = (
                        oracle_states[layer_index]["post_attention_state"]
                        - target_inputs.pop(layer_index)
                    )
                    return replace_last_token(output, required_attention)

                handles.append(
                    layer.register_forward_pre_hook(
                        layer_pre_hook, with_kwargs=True
                    )
                )
                handles.append(
                    layer.self_attn.register_forward_hook(
                        attention_hook, with_kwargs=True
                    )
                )

            else:
                def block_hook(
                    module, args, kwargs, output, layer_index=layer_index
                ):
                    return replace_last_token(
                        output, oracle_states[layer_index]["block_output"]
                    )

                handles.append(
                    layer.register_forward_hook(block_hook, with_kwargs=True)
                )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            target_inputs.clear()


def longest_common_suffix(left, right):
    count = 0
    limit = min(len(left), len(right))
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


def aligned_target_positions(
    target_prompt_length,
    full_prompt_length,
    generated_length,
    common_suffix_length,
    device,
):
    total = target_prompt_length + generated_length
    positions = torch.arange(total, device=device, dtype=torch.long)
    suffix_start = target_prompt_length - common_suffix_length
    gap = full_prompt_length - target_prompt_length
    positions[suffix_start:] += gap
    return positions.unsqueeze(0)


def regular_positions(length, device):
    return torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)

