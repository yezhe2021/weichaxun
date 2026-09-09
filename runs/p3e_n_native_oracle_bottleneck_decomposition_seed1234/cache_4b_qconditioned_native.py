import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import file_sha256, write_json
from p3e_b_common import SenderNativeHeadwiseCache
from p3e_l_common import ConditionedNativeCache


CONDITIONS = (
    "correct_question",
    "correct_question_hard_shuffled_evidence",
)


class NativeCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = list(layers)
        self.states = {}
        self.handles = []

    def __enter__(self):
        for layer_index in self.layers:
            attention = self.model.model.layers[layer_index].self_attn

            def hook(module, args, kwargs, layer_index=layer_index):
                hidden = args[0] if args else kwargs["hidden_states"]
                shape = (*hidden.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                values = module.v_proj(hidden).view(shape).transpose(1, 2)
                self.states[layer_index] = (keys.detach(), values.detach())

            self.handles.append(
                attention.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def question_prefix(tokenizer, question):
    return tokenizer(
        f"QUESTION\n{question}\n\n", add_special_tokens=False
    ).input_ids


def capture_evidence(model, capture, prefix_ids, evidence_ids, layers, device):
    ids = torch.tensor(
        [list(prefix_ids) + list(evidence_ids)], dtype=torch.long, device=device
    )
    capture.states.clear()
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    start, end = len(prefix_ids), ids.shape[1]
    keys, values = [], []
    for layer_index in layers:
        key, value = capture.states[layer_index]
        keys.append(key[0, :, start:end, :].transpose(0, 1).float().cpu())
        values.append(value[0, :, start:end, :].transpose(0, 1).float().cpu())
    return torch.stack(keys).half(), torch.stack(values).half()


def metadata(source, prefix_length):
    result = dict(source["metadata"])
    result.update(
        {
            "prefix_token_count": int(prefix_length),
            "evidence_token_count": int(source["keys"].shape[1]),
            "question_tokens_transmitted": 0,
            "evidence_only_slice": True,
            "pre_rope_keys": True,
            "native_model": "Qwen3-4B",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-memory", required=True)
    parser.add_argument("--mapping-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    base = SenderNativeHeadwiseCache(args.base_memory, capacity=4)
    mapping = ConditionedNativeCache(args.mapping_index, capacity=1)
    total = min(args.max_samples or len(base), len(base), len(mapping))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    layers = list(base.index["original_layer_indices"])
    entries = []

    with NativeCapture(model, layers) as capture, torch.inference_mode():
        for index in tqdm(range(total), desc="p3e_n_cache_4b_qconditioned"):
            current = base.load(index)
            map_entry = mapping.entries[index]
            if map_entry["id"] != current["row"]["id"]:
                raise RuntimeError("Hard-shuffled mapping is not sample-aligned")
            hard_index = int(map_entry["hard_evidence_index"])
            wrong = base.load(hard_index)
            prefix = question_prefix(tokenizer, current["row"]["question"])
            states = {}
            for condition, source in (
                ("correct_question", current),
                ("correct_question_hard_shuffled_evidence", wrong),
            ):
                evidence_ids = source["metadata"]["token_ids"]
                keys, values = capture_evidence(
                    model, capture, prefix, evidence_ids, layers, device
                )
                if keys.shape != (
                    len(layers),
                    len(evidence_ids),
                    8,
                    128,
                ):
                    raise RuntimeError(
                        f"Unexpected Qwen3-4B Native KV shape {tuple(keys.shape)}"
                    )
                states[condition] = {
                    "keys": keys,
                    "values": values,
                    "metadata": metadata(source, len(prefix)),
                    "source_id": source["row"]["id"],
                    "source_answer": source["row"]["answer"],
                }
            filename = f"sample_{index:05d}.pt"
            torch.save(
                {
                    "row": current["row"],
                    "conditions": states,
                    "hard_evidence_id": wrong["row"]["id"],
                },
                output / filename,
            )
            entries.append(
                {
                    "id": current["row"]["id"],
                    "file": filename,
                    "hard_evidence_index": hard_index,
                }
            )

    result = {
        "status": "complete",
        "experiment": "P3-E-N Qwen3-4B Q-conditioned Native KV cache",
        "samples": total,
        "entries": entries,
        "conditions": list(CONDITIONS),
        "shape": "[16,T,8,128]",
        "sender": args.model,
        "sender_config_sha256": file_sha256(Path(args.model) / "config.json"),
        "question_tokens_transmitted": 0,
        "receiver_sees_question_only": True,
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
