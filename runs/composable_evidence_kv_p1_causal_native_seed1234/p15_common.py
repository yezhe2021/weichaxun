import re

import torch
import torch.nn as nn

from causal_common import ResidualWriter


class IterativeMemoryReader(nn.Module):
    def __init__(self, memory_dim, state_dim, heads, rounds):
        super().__init__()
        self.memory_dim = memory_dim
        self.state_dim = state_dim
        self.rounds = rounds
        self.question_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, state_dim))
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.source_embedding = nn.Parameter(torch.zeros(2, memory_dim))
        self.null_memory = nn.Parameter(torch.zeros(2, 1, memory_dim))
        self.round_embedding = nn.Parameter(torch.zeros(rounds, state_dim))
        self.attention = nn.MultiheadAttention(
            state_dim, heads, kdim=memory_dim, vdim=memory_dim, batch_first=True
        )
        self.state_update = nn.GRUCell(state_dim, state_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        nn.init.normal_(self.source_embedding, std=0.02)
        nn.init.normal_(self.null_memory, std=0.02)
        nn.init.normal_(self.round_embedding, std=0.02)

    def _source(self, memory, source, batch_size, device):
        if memory is None:
            memory = self.null_memory[source].expand(batch_size, -1, -1)
        else:
            memory = memory.float()
        return self.memory_norm(memory) + self.source_embedding[source].view(1, 1, -1)

    def forward(self, question_state, memory_a, memory_b, return_rounds=False):
        batch_size = question_state.shape[0]
        memory = torch.cat(
            [
                self._source(memory_a, 0, batch_size, question_state.device),
                self._source(memory_b, 1, batch_size, question_state.device),
            ],
            dim=1,
        )
        state = self.question_projection(question_state.float())
        states = []
        attentions = []
        for round_index in range(self.rounds):
            query = state + self.round_embedding[round_index]
            readout, attention = self.attention(
                query.unsqueeze(1), memory, memory, need_weights=True, average_attn_weights=False
            )
            state = self.state_norm(self.state_update(readout[:, 0], state))
            state = state + self.ffn(state)
            states.append(state)
            attentions.append(attention)
        if return_rounds:
            return state, states, attentions
        return state


class GeneralEvidenceAdapter(nn.Module):
    def __init__(
        self,
        memory_dim,
        receiver_dim,
        state_dim,
        reader_heads,
        reader_rounds,
        writer_layers,
        writer_bottleneck=256,
        max_gate=0.5,
    ):
        super().__init__()
        self.writer_layers = tuple(int(layer) for layer in writer_layers)
        self.reader = IterativeMemoryReader(memory_dim, state_dim, reader_heads, reader_rounds)
        self.writers = nn.ModuleDict(
            {
                str(layer): ResidualWriter(receiver_dim, state_dim, writer_bottleneck, max_gate)
                for layer in self.writer_layers
            }
        )


def render_student_prompt(tokenizer, row):
    system = (
        "Answer the question using the external evidence available to you. Do not invent facts. "
        "If the external evidence does not support an answer, return INSUFFICIENT. "
        "End with exactly one line: FINAL: <answer_identifier>."
    )
    user = f"QUESTION\n{row['question']}"
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nFINAL:"


def render_teacher_prompt(tokenizer, row):
    system = (
        "Answer using only the supplied evidence. Combine any relevant facts as needed. "
        "End with exactly one line: FINAL: <answer_identifier>."
    )
    user = (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE\n{row['evidence_a']}\n{row['evidence_b']}"
    )
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nFINAL:"


def pack_answer(tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(f"FINAL: {answer}" + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    suffix_ids = suffix_ids[: max(1, max_length - len(prompt_ids))]
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels, len(prompt_ids)


def answer_token_view(logits, labels):
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    return logits[:, :-1, :][mask], shifted_labels[mask]


def normalize_answer(text):
    return re.sub(r"^[\s`*\"']+|[\s`*\"'.,;:!?]+$", "", str(text)).upper()


def extract_answer(text, candidates):
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL)
    allowed = list(dict.fromkeys([*candidates, "INSUFFICIENT"]))
    mapping = {normalize_answer(value): value for value in allowed}
    pattern = re.compile(
        r"(?<![A-Z0-9_])(" + "|".join(sorted(map(re.escape, mapping), key=len, reverse=True)) + r")(?![A-Z0-9_])",
        re.IGNORECASE,
    )
    anchored = re.findall(r"(?:FINAL|ANSWER|答案)\s*[:：]\s*([^\n\r]+)", clean, flags=re.IGNORECASE)
    for region in reversed(anchored):
        found = pattern.findall(region)
        if found:
            return mapping[normalize_answer(found[-1])], "final_anchor"
    found = pattern.findall(clean)
    if found:
        return mapping[normalize_answer(found[-1])], "last_valid_candidate"
    return "", "not_found"
