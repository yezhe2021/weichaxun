import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p3d3_common import extract_prediction
from p3e_f_common import read_json, read_jsonl


class PairedCanonicalTokenCache:
    def __init__(self, canonical_index, native_index, data_path, capacity=2):
        self.canonical_root = Path(canonical_index).parent
        self.native_root = Path(native_index).parent
        self.canonical_index = read_json(canonical_index)
        self.native_index = read_json(native_index)
        self.rows = read_jsonl(data_path)
        canonical_entries = self.canonical_index["entries"][:len(self.rows)]
        native_entries = self.native_index["entries"][:len(self.rows)]
        if len(canonical_entries) != len(self.rows) or len(native_entries) != len(self.rows):
            raise RuntimeError("Paired cache is shorter than the data split")
        for index, row in enumerate(self.rows):
            if canonical_entries[index]["id"] != row["id"] or native_entries[index]["id"] != row["id"]:
                raise RuntimeError(f"Canonical/Native/data ID mismatch at {index}")
        self.canonical_entries = canonical_entries
        self.native_entries = native_entries
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _resolve(root, entry):
        path = Path(entry["file"])
        return path if path.is_absolute() else root / path

    def load(self, index):
        if index not in self.loaded:
            canonical = torch.load(
                self._resolve(self.canonical_root, self.canonical_entries[index]),
                map_location="cpu", weights_only=False,
            )
            native = torch.load(
                self._resolve(self.native_root, self.native_entries[index]),
                map_location="cpu", weights_only=False,
            )
            row = self.rows[index]
            if canonical["id"] != row["id"] or native["row"]["id"] != row["id"]:
                raise RuntimeError(f"Payload ID mismatch at {index}")
            keys, values = canonical["keys"], canonical["values"]
            token_ids = native["metadata"]["token_ids"]
            valid = torch.as_tensor(native["metadata"]["valid_mask"], dtype=torch.bool)
            if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 16:
                raise RuntimeError(f"Invalid Canonical shape {tuple(keys.shape)}")
            if keys.shape[-2:] != (16, 128):
                raise RuntimeError(f"Expected Canonical heads [16,128], got {tuple(keys.shape[-2:])}")
            if keys.shape[1] != len(token_ids) or valid.numel() != len(token_ids):
                raise RuntimeError("Canonical token axis does not match Native token IDs")
            if canonical["mask"].numel() != len(token_ids):
                raise RuntimeError("Canonical mask length mismatch")
            if not torch.equal(canonical["mask"].bool(), valid):
                raise RuntimeError("Canonical and Native valid masks differ")
            self.loaded[index] = {
                "row": row,
                "keys": keys,
                "values": values,
                "mask": valid,
                "support_mask": canonical["support_mask"].bool(),
                "token_ids": torch.as_tensor(token_ids, dtype=torch.long),
                "evidence": native["evidence"],
                "offsets": native["metadata"]["offsets"],
            }
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, value):
        scale = value.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value.float() * scale * self.weight


class SoftPrefixDecoder(nn.Module):
    def __init__(
        self,
        layers=16,
        canonical_heads=16,
        head_dim=128,
        fusion_dim=128,
        hidden_size=512,
        receiver_dim=2560,
        transformer_layers=2,
        transformer_heads=8,
        ffn_dim=2048,
        max_tokens=1024,
        target_rms=1.0,
    ):
        super().__init__()
        self.layers = int(layers)
        self.canonical_heads = int(canonical_heads)
        self.head_dim = int(head_dim)
        self.fusion_dim = int(fusion_dim)
        self.hidden_size = int(hidden_size)
        self.receiver_dim = int(receiver_dim)
        self.max_tokens = int(max_tokens)
        self.k_norm = RMSNorm(head_dim)
        self.v_norm = RMSNorm(head_dim)
        self.kv_fusion = nn.Sequential(
            nn.Linear(head_dim * 2, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.head_queries = nn.Parameter(torch.empty(layers, fusion_dim))
        self.layer_embeddings = nn.Parameter(torch.empty(layers, fusion_dim))
        self.layer_score = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, 1),
        )
        self.layer_value = nn.Linear(fusion_dim, hidden_size)
        self.position_embeddings = nn.Parameter(torch.empty(max_tokens, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = RMSNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, receiver_dim)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)), persistent=True)
        nn.init.normal_(self.head_queries, std=0.02)
        nn.init.normal_(self.layer_embeddings, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.01)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def set_target_rms(self, value):
        self.target_rms.fill_(float(value))

    def forward(self, keys, values, mask, return_diagnostics=False):
        if keys.shape != values.shape:
            raise RuntimeError("Canonical K/V shapes differ")
        expected = (self.layers, self.canonical_heads, self.head_dim)
        if keys.ndim != 4 or (keys.shape[0], keys.shape[2], keys.shape[3]) != expected:
            raise RuntimeError(f"Expected [16,T,16,128], got {tuple(keys.shape)}")
        tokens = keys.shape[1]
        if tokens > self.max_tokens:
            raise RuntimeError(f"Evidence length {tokens} exceeds decoder max_tokens={self.max_tokens}")
        mask = mask.to(device=keys.device, dtype=torch.bool)
        if mask.numel() != tokens or not mask.any():
            raise RuntimeError("Invalid Soft Prefix token mask")
        fused = self.kv_fusion(torch.cat((self.k_norm(keys), self.v_norm(values)), dim=-1))
        head_scores = torch.einsum("lthd,ld->lth", fused, self.head_queries)
        head_weights = head_scores.softmax(dim=-1)
        layer_tokens = torch.einsum("lth,lthd->ltd", head_weights, fused)
        layer_context = layer_tokens + self.layer_embeddings[:, None, :]
        layer_scores = self.layer_score(layer_context).squeeze(-1)
        layer_weights = layer_scores.softmax(dim=0)
        mixed = torch.einsum("lt,ltd->td", layer_weights, self.layer_value(layer_context))
        mixed = mixed + self.position_embeddings[:tokens]
        mixed = self.token_mixer(
            mixed.unsqueeze(0), src_key_padding_mask=(~mask).unsqueeze(0)
        ).squeeze(0)
        output = self.output_projection(self.output_norm(mixed))
        rms = output.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        output = output.float() / rms * self.target_rms.float()
        output = output * mask[:, None]
        if return_diagnostics:
            return output, {
                "head_weights": head_weights.detach(),
                "layer_weights": layer_weights.detach(),
                "pre_scale_rms": rms.detach().squeeze(-1),
            }
        return output

    def metadata(self):
        return {
            "layers": self.layers,
            "canonical_heads": self.canonical_heads,
            "head_dim": self.head_dim,
            "fusion_dim": self.fusion_dim,
            "hidden_size": self.hidden_size,
            "receiver_dim": self.receiver_dim,
            "transformer_layers": len(self.token_mixer.layers),
            "transformer_heads": self.token_mixer.layers[0].self_attn.num_heads,
            "ffn_dim": self.token_mixer.layers[0].linear1.out_features,
            "max_tokens": self.max_tokens,
            "target_rms": float(self.target_rms),
            "token_axis_preserved": True,
            "slot_compression": False,
            "question_conditioned": False,
        }


def evidence_embedding_rms(model, cache, count, device):
    total, tokens = torch.zeros((), device=device), 0
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        for index in range(count):
            payload = cache.load(index)
            ids = payload["token_ids"].to(device)
            valid = payload["mask"].to(device)
            values = embedding(ids)[valid].float()
            total += values.square().mean(dim=-1).sqrt().sum()
            tokens += values.shape[0]
    if tokens == 0:
        raise RuntimeError("No valid Evidence tokens for RMS calibration")
    return float(total / tokens)


def verify_tokenizer_alignment(tokenizer, payload):
    encoded = tokenizer(
        payload["evidence"], add_special_tokens=True,
        truncation=True, max_length=1024,
    ).input_ids
    expected = payload["token_ids"].tolist()
    if encoded != expected:
        raise RuntimeError("Qwen3-4B tokenizer does not reproduce cached Qwen3-8B Evidence token IDs")


def template_token_ids(tokenizer, row, evidence_ids=None, include_answer=True):
    prefix = tokenizer("Evidence:\n", add_special_tokens=False).input_ids
    middle = tokenizer("\n\nQuestion:\n", add_special_tokens=False).input_ids
    question = tokenizer(row["question"], add_special_tokens=False).input_ids
    answer_prefix = tokenizer("\n\nAnswer:\nFINAL:", add_special_tokens=False).input_ids
    answer = tokenizer(
        " " + row["answer"] + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    ).input_ids if include_answer else []
    evidence = [] if evidence_ids is None else list(evidence_ids)
    ids = prefix + evidence + middle + question + answer_prefix + answer
    evidence_slice = (len(prefix), len(prefix) + len(evidence))
    question_start = evidence_slice[1] + len(middle)
    question_slice = (question_start, question_start + len(question))
    answer_start = question_slice[1] + len(answer_prefix)
    answer_slice = (answer_start, answer_start + len(answer))
    return {
        "ids": ids,
        "prefix": prefix,
        "middle": middle,
        "question": question,
        "answer_prefix": answer_prefix,
        "answer": answer,
        "evidence_slice": evidence_slice,
        "question_slice": question_slice,
        "answer_slice": answer_slice,
    }


def build_teacher_forcing_batch(
    model, tokenizer, row, evidence_ids, soft_evidence, device,
):
    layout = template_token_ids(tokenizer, row, evidence_ids.tolist(), include_answer=True)
    ids = torch.tensor(layout["ids"], dtype=torch.long, device=device)
    embeddings = model.get_input_embeddings()(ids)
    left, right = layout["evidence_slice"]
    if soft_evidence.shape != embeddings[left:right].shape:
        raise RuntimeError("Soft Prefix shape does not match exact Evidence embedding span")
    student_embeddings = torch.cat(
        (embeddings[:left], soft_evidence.to(embeddings.dtype), embeddings[right:]), dim=0
    )
    labels = ids.clone()
    labels[:layout["answer_slice"][0]] = -100
    attention_mask = torch.ones(1, ids.numel(), dtype=torch.long, device=device)
    position_ids = torch.arange(ids.numel(), device=device).unsqueeze(0)
    question_mask = torch.zeros(ids.numel(), dtype=torch.bool, device=device)
    question_mask[slice(*layout["question_slice"])] = True
    return {
        "ids": ids.unsqueeze(0),
        "teacher_embeddings": embeddings.unsqueeze(0),
        "student_embeddings": student_embeddings.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "question_mask": question_mask,
        "layout": layout,
    }


def answer_mean_nll(logits, labels):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    selected = shifted_labels != -100
    if not selected.any():
        raise RuntimeError("No answer tokens selected")
    return F.cross_entropy(shifted_logits[selected], shifted_labels[selected], reduction="mean")


def answer_kl(student_logits, teacher_logits, labels, temperature=2.0):
    selected = labels[:, 1:] != -100
    student = student_logits[:, :-1].float()[selected] / temperature
    teacher = teacher_logits[:, :-1].float()[selected].detach() / temperature
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def hidden_alignment_loss(student_hidden, teacher_hidden, question_mask, layer_indices):
    losses = []
    for layer_index in layer_indices:
        student = student_hidden[layer_index + 1][0, question_mask].float()
        teacher = teacher_hidden[layer_index + 1][0, question_mask].float().detach()
        cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
        student_norm = F.layer_norm(student, (student.shape[-1],))
        teacher_norm = F.layer_norm(teacher, (teacher.shape[-1],))
        losses.append(0.5 * cosine + 0.5 * F.mse_loss(student_norm, teacher_norm))
    return torch.stack(losses).mean()


def embedding_reconstruction_loss(soft, target, mask):
    selected_soft = soft[mask].float()
    selected_target = target[mask].float().detach()
    cosine = 1.0 - F.cosine_similarity(selected_soft, selected_target, dim=-1).mean()
    mse = F.mse_loss(selected_soft, selected_target)
    return cosine + 0.1 * mse, {"cosine_loss": cosine, "mse": mse}


def assert_optimizer_only_decoder(model, decoder, optimizer, extra_frozen=()):
    allowed = {id(parameter) for parameter in decoder.parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if allowed != actual:
        raise RuntimeError("Optimizer must contain exactly Soft Prefix Decoder parameters")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver parameter is trainable")
    for module in extra_frozen:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("A frozen auxiliary module is trainable")


def assert_frozen_gradients(model, extra_frozen=()):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Gradient reached frozen Receiver parameter")
    for module in extra_frozen:
        if any(parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError("Gradient reached a frozen auxiliary module")


def decoder_checkpoint(decoder, stage, epoch, history, args):
    return {
        "decoder": {name: value.detach().cpu() for name, value in decoder.state_dict().items()},
        "decoder_metadata": decoder.metadata(),
        "stage": stage,
        "epoch": int(epoch),
        "history": history,
        "args": vars(args),
    }


def load_decoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["decoder_metadata"]
    decoder = SoftPrefixDecoder(
        layers=metadata["layers"],
        canonical_heads=metadata["canonical_heads"],
        head_dim=metadata["head_dim"],
        fusion_dim=metadata["fusion_dim"],
        hidden_size=metadata["hidden_size"],
        receiver_dim=metadata["receiver_dim"],
        transformer_layers=metadata["transformer_layers"],
        transformer_heads=metadata["transformer_heads"],
        ffn_dim=metadata["ffn_dim"],
        max_tokens=metadata["max_tokens"],
        target_rms=metadata["target_rms"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    return decoder, checkpoint


def _manual_prefill(model, embeddings, attention_mask):
    position_ids = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0)
    return model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def manual_greedy_generate(
    model, tokenizer, prompt_embeddings, max_new_tokens,
):
    attention_mask = torch.ones(
        1, prompt_embeddings.shape[1], dtype=torch.long, device=prompt_embeddings.device
    )
    output = _manual_prefill(model, prompt_embeddings, attention_mask)
    generated = []
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)),
            dim=1,
        )
        position_ids = torch.tensor(
            [[attention_mask.shape[1] - 1]], dtype=torch.long, device=attention_mask.device
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    prediction, method = extract_prediction("FINAL:" + text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": generated,
        "eos_reached": tokenizer.eos_token_id in generated,
    }


@torch.inference_mode()
def manual_greedy_generate_ids(model, tokenizer, prompt_ids, max_new_tokens):
    prompt_ids = prompt_ids.to(model.device).unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids.to(model.device)
    attention_mask = torch.ones_like(prompt_ids)
    position_ids = torch.arange(prompt_ids.shape[1], device=model.device).unsqueeze(0)
    output = model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    generated = []
    past = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            (attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=model.device)),
            dim=1,
        )
        position_ids = torch.tensor(
            [[attention_mask.shape[1] - 1]], dtype=torch.long, device=model.device
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    prediction, method = extract_prediction("FINAL:" + text)
    return {
        "text": text,
        "prediction": prediction,
        "parse_method": method,
        "token_ids": generated,
        "eos_reached": tokenizer.eos_token_id in generated,
    }


def prompt_embeddings(model, tokenizer, row, evidence_embeddings=None):
    evidence_ids = [] if evidence_embeddings is None else [0] * evidence_embeddings.shape[0]
    layout = template_token_ids(tokenizer, row, evidence_ids, include_answer=False)
    left, right = layout["evidence_slice"]
    real_ids = layout["prefix"] + layout["middle"] + layout["question"] + layout["answer_prefix"]
    real_embeddings = model.get_input_embeddings()(
        torch.tensor(real_ids, dtype=torch.long, device=model.device)
    )
    prefix_count = len(layout["prefix"])
    if evidence_embeddings is None:
        return real_embeddings.unsqueeze(0)
    return torch.cat(
        (
            real_embeddings[:prefix_count],
            evidence_embeddings.to(real_embeddings.dtype),
            real_embeddings[prefix_count:],
        ),
        dim=0,
    ).unsqueeze(0)


def exact_prompt_ids(tokenizer, row, evidence_ids):
    return torch.tensor(
        template_token_ids(tokenizer, row, evidence_ids, include_answer=False)["ids"],
        dtype=torch.long,
    )


def nearest_token_metrics(predicted, token_ids, mask, embedding_weight, max_positions=512):
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() > max_positions:
        selected = torch.linspace(
            0, positions.numel() - 1, max_positions, device=positions.device
        ).round().long()
        positions = positions[selected]
    queries = F.normalize(predicted[positions].float(), dim=-1)
    vocabulary = F.normalize(embedding_weight.detach().float(), dim=-1)
    top1, top5, total = 0, 0, 0
    for start in range(0, queries.shape[0], 16):
        batch = queries[start:start + 16]
        indices = (batch @ vocabulary.T).topk(5, dim=-1).indices
        target = token_ids[positions[start:start + 16]].to(indices.device)
        top1 += int((indices[:, 0] == target).sum())
        top5 += int((indices == target[:, None]).any(dim=-1).sum())
        total += target.numel()
    return {
        "sampled_positions": total,
        "nearest_token_top1": top1 / max(total, 1),
        "nearest_token_top5": top5 / max(total, 1),
    }


def reconstruction_metrics(predicted, target, token_ids, mask, embedding_weight=None):
    selected_predicted = predicted[mask].float()
    selected_target = target[mask].float()
    cosine_values = F.cosine_similarity(selected_predicted, selected_target, dim=-1)
    squared = (selected_predicted - selected_target).square().mean(dim=-1)
    result = {
        "tokens": int(mask.sum()),
        "embedding_cosine": float(cosine_values.mean()),
        "embedding_mse": float(squared.mean()),
        "position_quartiles": {},
    }
    valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
    for quartile in range(4):
        left = math.floor(valid_positions.numel() * quartile / 4)
        right = math.floor(valid_positions.numel() * (quartile + 1) / 4)
        chosen = valid_positions[left:right]
        if chosen.numel():
            result["position_quartiles"][str(quartile)] = {
                "cosine": float(F.cosine_similarity(
                    predicted[chosen].float(), target[chosen].float(), dim=-1
                ).mean()),
                "mse": float((predicted[chosen].float() - target[chosen].float()).square().mean()),
            }
    if embedding_weight is not None:
        result.update(nearest_token_metrics(
            predicted, token_ids, mask, embedding_weight,
        ))
    return result
