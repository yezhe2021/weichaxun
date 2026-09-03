import math
from contextlib import contextmanager

import torch
import torch.nn as nn

from p3e_p_common import SELECTED_LAYERS, SharedStateCorrector


class UncompressedMultiSlotDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=2560,
        slots=8,
        slot_dim=256,
        kv_heads=8,
        head_dim=128,
        rounds=2,
    ):
        super().__init__()
        self.slots = slots
        self.slot_dim = slot_dim
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rounds = rounds
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.slot_seed = nn.Linear(hidden_size, slots * slot_dim)
        self.slot_identity = nn.Parameter(torch.zeros(1, 1, slots, slot_dim))
        self.slot_norms = nn.ModuleList(
            [nn.RMSNorm(slot_dim) for _ in range(rounds)]
        )
        self.query_projections = nn.ModuleList(
            [
                nn.Linear(slot_dim, kv_heads * head_dim)
                for _ in range(rounds)
            ]
        )
        self.head_merges = nn.ModuleList(
            [
                nn.Linear(kv_heads * head_dim, slot_dim)
                for _ in range(rounds)
            ]
        )
        self.slot_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(slot_dim),
                    nn.Linear(slot_dim, slot_dim * 4),
                    nn.SiLU(),
                    nn.Linear(slot_dim * 4, slot_dim),
                )
                for _ in range(rounds)
            ]
        )
        self.state_query = nn.Linear(hidden_size, slot_dim)
        self.slot_key = nn.Linear(slot_dim, slot_dim)
        self.slot_value = nn.Linear(slot_dim, slot_dim)

    def memory_round(self, slots, keys, values, mask, round_index):
        queries = self.query_projections[round_index](
            self.slot_norms[round_index](slots)
        ).reshape(
            *slots.shape[:-1], self.kv_heads, self.head_dim
        )
        scores = torch.einsum(
            "bskhd,thd->bskht", queries, keys.float()
        ) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask[None, None, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attention = scores.softmax(dim=-1)
        headwise = torch.einsum(
            "bskht,thd->bskhd", attention, values.float()
        )
        update = self.head_merges[round_index](
            headwise.reshape(*headwise.shape[:-2], self.kv_heads * self.head_dim)
        )
        slots = slots + update
        slots = slots + self.slot_mlps[round_index](slots)
        return slots, attention

    def forward(self, hidden, keys, values, mask):
        normalized_hidden = self.hidden_norm(hidden.float())
        slots = self.slot_seed(normalized_hidden).reshape(
            *hidden.shape[:-1], self.slots, self.slot_dim
        )
        slots = slots + self.slot_identity
        attentions = []
        for round_index in range(self.rounds):
            slots, attention = self.memory_round(
                slots, keys, values, mask, round_index
            )
            attentions.append(attention)

        # The eight slots remain independent until the current Receiver state
        # queries them. There is no content-independent slot pooling.
        query = self.state_query(normalized_hidden)
        slot_keys = self.slot_key(slots)
        slot_values = self.slot_value(slots)
        slot_scores = torch.einsum(
            "bsd,bskd->bsk", query, slot_keys
        ) / math.sqrt(self.slot_dim)
        slot_attention = slot_scores.softmax(dim=-1)
        correction_representation = torch.einsum(
            "bsk,bskd->bsd", slot_attention, slot_values
        )
        return correction_representation, slots, attentions, slot_attention


class C2TrajectoryReader(nn.Module):
    def __init__(self, model, selected_layers=SELECTED_LAYERS):
        super().__init__()
        self.selected_layers = list(selected_layers)
        self.decoder = UncompressedMultiSlotDecoder()
        self.corrector = SharedStateCorrector(
            memory_dim=256,
            layers=len(self.selected_layers),
            hidden_size=int(model.config.hidden_size),
            width=256,
        )
        self._layer_inputs = {}

    @contextmanager
    def inject(self, model, memory, trace_positions=None, trace=None):
        handles = []
        for local, layer_index in enumerate(self.selected_layers):
            layer = model.model.layers[layer_index]

            def layer_pre_hook(module, args, kwargs, local=local):
                self._layer_inputs[local] = args[0]

            def attention_hook(
                module, args, kwargs, output, local=local, layer_index=layer_index
            ):
                native_output = output[0] if isinstance(output, tuple) else output
                if local not in self._layer_inputs:
                    raise RuntimeError("Layer input was not captured")
                post_attention = self._layer_inputs.pop(local) + native_output
                representation, slots, rounds, slot_attention = self.decoder(
                    post_attention,
                    memory["keys"][local],
                    memory["values"][local],
                    memory["mask"],
                )
                if slots.shape[-2:] != (8, 256):
                    raise RuntimeError(
                        f"C2 did not preserve eight 256D slots: {tuple(slots.shape)}"
                    )
                delta = self.corrector(
                    post_attention, representation, local
                ).to(native_output.dtype)
                corrected = post_attention + delta
                if trace is not None and trace_positions is not None:
                    trace[layer_index] = {
                        "original": post_attention[:, trace_positions].float(),
                        "corrected": corrected[:, trace_positions].float(),
                        "slots": slots[:, trace_positions].float(),
                        "slot_attention": slot_attention[
                            :, trace_positions
                        ].float(),
                    }
                patched = native_output + delta
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

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
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._layer_inputs.clear()

    def metadata(self):
        return {
            "mode": "c2_uncompressed_multislot",
            "selected_layers": self.selected_layers,
            "trajectory_target": "receiver_post_attention_state",
            "native_memory": "[16,T,8,128]",
            "memory_rounds": 2,
            "slots": 8,
            "slot_dim": 256,
            "slot_pooling_before_state_query": False,
            "final_read": "receiver_hidden_queries_all_8_slots",
        }


def load_c2_reader(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    reader = C2TrajectoryReader(
        model, checkpoint["reader_metadata"]["selected_layers"]
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.eval()
    return reader, checkpoint

