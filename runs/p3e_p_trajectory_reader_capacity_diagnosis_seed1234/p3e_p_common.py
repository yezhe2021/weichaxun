import math
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3d3_common import (
    apply_chat,
    evidence_block,
    extract_prediction,
    read_json,
)
from p3e_a_common import PerQueryHeadResidual


SELECTED_LAYERS = [0, 2, 5, 7, 9, 12, 14, 16, 19, 21, 23, 26, 28, 30, 33, 35]
MODES = ("a0", "a1", "c1")


def question_prompt(tokenizer, row):
    system = "Answer the question with a short answer. End with exactly FINAL: <answer>."
    return apply_chat(tokenizer, system, f"QUESTION\n{row['question']}") + "FINAL:"


def full_text_prompt(tokenizer, row):
    system = (
        "Answer the question using the supplied gold evidence. "
        "Give a short answer. End with exactly FINAL: <answer>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"GOLD SUPPORTING EVIDENCE\n{evidence_block(row)}"
    )
    return apply_chat(tokenizer, system, user) + "FINAL:"


def answer_suffix(tokenizer, answer):
    return tokenizer(
        " " + answer + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    ).input_ids


def common_suffix_length(left, right):
    count = 0
    while count < min(len(left), len(right)) and left[-1-count] == right[-1-count]:
        count += 1
    return count


def build_position_ids(target_prompt_ids, full_prompt_ids, suffix_length, device):
    common = common_suffix_length(target_prompt_ids, full_prompt_ids)
    if common == 0:
        raise RuntimeError("Teacher and student prompts have no common answer suffix")
    total = len(target_prompt_ids) + suffix_length
    positions = torch.arange(total, device=device, dtype=torch.long)
    positions[len(target_prompt_ids) - common:] += (
        len(full_prompt_ids) - len(target_prompt_ids)
    )
    return positions.unsqueeze(0), common


def prediction_positions(prompt_length, suffix_length, device):
    return torch.arange(
        prompt_length - 1,
        prompt_length + suffix_length - 1,
        device=device,
        dtype=torch.long,
    )


def pack_student(tokenizer, row, device):
    target_prompt_ids = tokenizer(
        question_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    full_prompt_ids = tokenizer(
        full_text_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    suffix = answer_suffix(tokenizer, row["answer"])
    ids = torch.tensor(
        [target_prompt_ids + suffix], dtype=torch.long, device=device
    )
    labels = ids.clone()
    labels[:, :len(target_prompt_ids)] = -100
    position_ids, common = build_position_ids(
        target_prompt_ids, full_prompt_ids, len(suffix), device
    )
    positions = prediction_positions(
        len(target_prompt_ids), len(suffix), device
    )
    teacher_positions = prediction_positions(
        len(full_prompt_ids), len(suffix), device
    )
    if not torch.equal(
        position_ids[0, positions], teacher_positions
    ):
        raise RuntimeError("Teacher/student answer prediction positions are not aligned")
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": position_ids,
        "labels": labels,
        "prediction_positions": positions,
        "answer_token_ids": suffix,
        "target_prompt_length": len(target_prompt_ids),
        "full_prompt_length": len(full_prompt_ids),
        "common_suffix_length": common,
    }


def answer_mean_nll(logits, labels):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    selected = shifted_labels != -100
    losses = F.cross_entropy(
        shifted_logits[selected], shifted_labels[selected], reduction="none"
    )
    if not losses.numel():
        raise RuntimeError("No answer labels")
    return losses.mean()


def rms_normalize(tensor, eps=1e-6):
    return tensor.float() / torch.sqrt(
        tensor.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )


def normalized_state_loss(corrected, teacher):
    return F.mse_loss(rms_normalize(corrected), rms_normalize(teacher))


def state_diagnostics(original, corrected, teacher):
    original_n = rms_normalize(original)
    corrected_n = rms_normalize(corrected)
    teacher_n = rms_normalize(teacher)
    return {
        "cosine_before": float(
            F.cosine_similarity(original_n, teacher_n, dim=-1).mean()
        ),
        "cosine_after": float(
            F.cosine_similarity(corrected_n, teacher_n, dim=-1).mean()
        ),
        "normalized_mse_before": float(F.mse_loss(original_n, teacher_n)),
        "normalized_mse_after": float(F.mse_loss(corrected_n, teacher_n)),
    }


class TeacherTrajectoryCache:
    def __init__(self, index_path, capacity=8):
        self.path = Path(index_path)
        self.root = self.path.parent
        self.index = read_json(index_path)
        self.entries = self.index["entries"]
        self.capacity = capacity
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(
                self.root / self.entries[index]["file"],
                map_location="cpu",
                weights_only=False,
            )
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


class SharedStateCorrector(nn.Module):
    def __init__(self, memory_dim, layers=16, hidden_size=2560, width=256):
        super().__init__()
        self.h_norm = nn.RMSNorm(hidden_size)
        self.z_norm = nn.RMSNorm(memory_dim)
        self.h_projection = nn.Linear(hidden_size, width)
        self.z_projection = nn.Linear(memory_dim, width)
        self.layer_embedding = nn.Embedding(layers, 64)
        self.layer_projection = nn.Linear(64, width, bias=False)
        self.output = nn.Linear(width, hidden_size)
        self.gates = nn.Parameter(torch.full((layers,), 0.01))

    def forward(self, hidden, memory, local_layer):
        layer = torch.full(
            hidden.shape[:-1],
            local_layer,
            dtype=torch.long,
            device=hidden.device,
        )
        fused = F.silu(
            self.h_projection(self.h_norm(hidden.float()))
            + self.z_projection(self.z_norm(memory.float()))
            + self.layer_projection(self.layer_embedding(layer))
        )
        return self.gates[local_layer] * self.output(fused)


class AVMemoryInterface(nn.Module):
    def __init__(self, model, selected_layers=SELECTED_LAYERS, rank=32):
        super().__init__()
        config = model.config
        self.selected_layers = list(selected_layers)
        self.query_heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.query_per_kv = self.query_heads // self.kv_heads
        self.query_adapters = nn.ModuleList(
            [
                PerQueryHeadResidual(self.query_heads, self.head_dim, rank)
                for _ in self.selected_layers
            ]
        )
        self.corrector = SharedStateCorrector(2560, len(self.selected_layers))

    def read(self, query, keys, values, mask, local, o_proj):
        if query.shape[1] == self.query_heads:
            query = query.transpose(1, 2)
        query = self.query_adapters[local](query)
        grouped = query.reshape(
            query.shape[0],
            query.shape[1],
            self.kv_heads,
            self.query_per_kv,
            self.head_dim,
        )
        scores = torch.einsum(
            "bshgd,thd->bshgt", grouped, keys.float()
        ) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask[None, None, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attention = scores.softmax(dim=-1)
        readout = torch.einsum(
            "bshgt,thd->bshgd", attention, values.float()
        )
        flattened = readout.reshape(
            query.shape[0], query.shape[1], self.query_heads * self.head_dim
        )
        return o_proj(flattened.to(o_proj.weight.dtype)), attention


class TwoStepKVDecoder(nn.Module):
    def __init__(self, hidden_size=2560, slots=4, kv_heads=8, head_dim=128):
        super().__init__()
        self.slots = slots
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.query_generator = nn.Linear(
            hidden_size, slots * kv_heads * head_dim
        )
        self.second_query = nn.Linear(head_dim, kv_heads * head_dim)
        self.slot_mlp = nn.Sequential(
            nn.RMSNorm(head_dim),
            nn.Linear(head_dim, head_dim * 4),
            nn.SiLU(),
            nn.Linear(head_dim * 4, head_dim),
        )
        self.head_logits = nn.Parameter(torch.zeros(2, kv_heads))
        self.pool_query = nn.Linear(hidden_size, head_dim)

    def attend(self, queries, keys, values, mask):
        scores = torch.einsum(
            "bsmhd,thd->bsmht", queries, keys.float()
        ) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask[None, None, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = scores.softmax(dim=-1)
        values_by_head = torch.einsum(
            "bsmht,thd->bsmhd", weights, values.float()
        )
        return values_by_head, weights

    def forward(self, hidden, keys, values, mask):
        normalized = self.hidden_norm(hidden.float())
        first_queries = self.query_generator(normalized).reshape(
            *hidden.shape[:-1], self.slots, self.kv_heads, self.head_dim
        )
        first_values, first_attention = self.attend(
            first_queries, keys, values, mask
        )
        first_slots = torch.einsum(
            "bsmhd,h->bsmd",
            first_values,
            self.head_logits[0].softmax(dim=-1),
        )
        first_slots = first_slots + self.slot_mlp(first_slots)
        second_queries = self.second_query(first_slots).reshape(
            *first_slots.shape[:-1], self.kv_heads, self.head_dim
        )
        second_values, second_attention = self.attend(
            second_queries, keys, values, mask
        )
        second_slots = first_slots + torch.einsum(
            "bsmhd,h->bsmd",
            second_values,
            self.head_logits[1].softmax(dim=-1),
        )
        second_slots = second_slots + self.slot_mlp(second_slots)
        pool_query = self.pool_query(normalized)
        pool_scores = torch.einsum(
            "bsd,bsmd->bsm", pool_query, second_slots
        ) / math.sqrt(self.head_dim)
        pooled = torch.einsum(
            "bsm,bsmd->bsd", pool_scores.softmax(dim=-1), second_slots
        )
        return pooled, (first_attention, second_attention)


class FullKVMemoryInterface(nn.Module):
    def __init__(self, model, selected_layers=SELECTED_LAYERS):
        super().__init__()
        self.selected_layers = list(selected_layers)
        self.decoder = TwoStepKVDecoder()
        self.corrector = SharedStateCorrector(128, len(self.selected_layers))


class TrajectoryReader(nn.Module):
    def __init__(self, model, mode, selected_layers=SELECTED_LAYERS):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.selected_layers = list(selected_layers)
        self.interface = (
            AVMemoryInterface(model, selected_layers)
            if mode in ("a0", "a1")
            else FullKVMemoryInterface(model, selected_layers)
        )
        self._queries = {}
        self._layer_inputs = {}

    @contextmanager
    def inject(self, model, memory, trace_positions=None, trace=None):
        handles = []
        for local, layer_index in enumerate(self.selected_layers):
            layer = model.model.layers[layer_index]

            def layer_pre_hook(module, args, kwargs, local=local):
                self._layer_inputs[local] = args[0]

            handles.append(
                layer.register_forward_pre_hook(
                    layer_pre_hook, with_kwargs=True
                )
            )
            if self.mode in ("a0", "a1"):
                def q_hook(module, args, output, local=local):
                    self._queries[local] = output

                handles.append(layer.self_attn.q_norm.register_forward_hook(q_hook))

            def attention_hook(
                module, args, kwargs, output, local=local, layer_index=layer_index
            ):
                native_output = output[0] if isinstance(output, tuple) else output
                if local not in self._layer_inputs:
                    raise RuntimeError("Layer input was not captured")
                post_attention = self._layer_inputs.pop(local) + native_output
                keys = memory["keys"][local]
                values = memory["values"][local]
                mask = memory["mask"]
                if self.mode in ("a0", "a1"):
                    if local not in self._queries:
                        raise RuntimeError("Native query was not captured")
                    external, _ = self.interface.read(
                        self._queries.pop(local),
                        keys,
                        values,
                        mask,
                        local,
                        module.o_proj,
                    )
                else:
                    external, _ = self.interface.decoder(
                        post_attention, keys, values, mask
                    )
                delta = self.interface.corrector(
                    post_attention, external, local
                ).to(native_output.dtype)
                corrected = post_attention + delta
                if trace is not None and trace_positions is not None:
                    trace[layer_index] = {
                        "original": post_attention[:, trace_positions].float(),
                        "corrected": corrected[:, trace_positions].float(),
                    }
                patched_output = native_output + delta
                if isinstance(output, tuple):
                    return (patched_output,) + output[1:]
                return patched_output

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
            self._queries.clear()
            self._layer_inputs.clear()

    def metadata(self):
        return {
            "mode": self.mode,
            "selected_layers": self.selected_layers,
            "trajectory_target": "receiver_post_attention_state",
            "position_alignment": "answer suffix aligned to full-text positions",
            "full_kv_rounds": 2 if self.mode == "c1" else None,
            "full_kv_latent_slots": 4 if self.mode == "c1" else None,
        }


def native_memory(payload, device):
    keys = payload["keys"].float().to(device)
    values = payload["values"].float().to(device)
    mask = torch.as_tensor(
        payload["metadata"]["valid_mask"], dtype=torch.bool, device=device
    )
    return {"keys": keys, "values": values, "mask": mask}


def stack_trace(trace, selected_layers, key):
    return torch.stack([trace[layer][key][0] for layer in selected_layers])


def load_reader(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    reader = TrajectoryReader(
        model,
        checkpoint["reader_metadata"]["mode"],
        checkpoint["reader_metadata"]["selected_layers"],
    ).to(device)
    reader.load_state_dict(checkpoint["reader"])
    reader.eval()
    return reader, checkpoint


@torch.inference_mode()
def generate_student(
    model,
    tokenizer,
    reader,
    row,
    memory,
    max_new_tokens,
    enabled=True,
):
    device = model.device
    target_prompt_ids = tokenizer(
        question_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    full_prompt_ids = tokenizer(
        full_text_prompt(tokenizer, row), add_special_tokens=False
    ).input_ids
    common = common_suffix_length(target_prompt_ids, full_prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        ids = torch.tensor(
            [target_prompt_ids + generated], dtype=torch.long, device=device
        )
        positions = torch.arange(ids.shape[1], device=device)
        positions[len(target_prompt_ids)-common:] += (
            len(full_prompt_ids) - len(target_prompt_ids)
        )
        kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        )
        if enabled:
            with reader.inject(model, memory):
                output = model(**kwargs)
        else:
            output = model(**kwargs)
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    prediction, method = extract_prediction(text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": generated,
        "eos_reached": tokenizer.eos_token_id in generated,
    }
