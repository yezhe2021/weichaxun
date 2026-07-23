import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_scores, generate, hard_negative_mapping, load_receiver, normalize_answer, seed_everything, write_json, write_jsonl
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import SenderNativeHeadwiseCache, load_writer, writer_memory
from p3e_c3_common import intervene_pair


MODES = ("both", "first_only", "second_only", "swap", "copy_first_to_second", "copy_second_to_first")


def trace_change(baseline, changed, pair, reader):
    first, second = 2 * pair, 2 * pair + 1; per_layer = []
    routes = reader.routes().detach().float().cpu()
    for local, layer in enumerate(reader.selected_layers):
        base, other = baseline[layer][0], changed[layer][0]; route = routes[local]
        selected = route[:, [first, second]].sum(-1) > 0
        if not selected.any(): selected = torch.ones(route.shape[0], dtype=torch.bool)
        base_attention = base["attention"].detach().float().cpu()[0, -1, selected][:, [first, second]]
        other_attention = other["attention"].detach().float().cpu()[0, -1, selected][:, [first, second]]
        midpoint = 0.5 * (base_attention + other_attention)
        js = 0.5 * ((base_attention.clamp_min(1e-8) * (base_attention.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1).mean() +
                    (other_attention.clamp_min(1e-8) * (other_attention.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(-1).mean())
        base_head = base["headwise_readout"].detach().float().cpu()[0, -1, selected]
        other_head = other["headwise_readout"].detach().float().cpu()[0, -1, selected]
        head_cosine = F.cosine_similarity(base_head, other_head, dim=-1).mean()
        projected_cosine = F.cosine_similarity(base["projected"].detach().float().cpu()[0, -1], other["projected"].detach().float().cpu()[0, -1], dim=-1)
        per_layer.append({"layer": layer, "selected_query_heads": selected.nonzero().flatten().tolist(), "attention_js": float(js),
                          "headwise_readout_cosine": float(head_cosine), "projected_readout_cosine": float(projected_cosine),
                          "projected_rms_ratio": float(other["projected"].detach().float().cpu()[0, -1].pow(2).mean().sqrt() / base["projected"].detach().float().cpu()[0, -1].pow(2).mean().sqrt().clamp_min(1e-8))})
    return per_layer


def summarize(rows, pair, mode, memory_condition):
    selected = [row for row in rows if row["pair_index"] == pair and row["mode"] == mode and row["memory_condition"] == memory_condition]
    result = {"n": len(selected), "em": sum(row["em"] for row in selected) / len(selected), "f1": sum(row["f1"] for row in selected) / len(selected), "by_type": {}}
    for kind in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == kind]
        if group: result["by_type"][kind] = {"n": len(group), "em": sum(row["em"] for row in group) / len(group), "f1": sum(row["f1"] for row in group) / len(group)}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--memory", required=True); parser.add_argument("--writer", required=True); parser.add_argument("--reader", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device(args.device); output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.memory); count = min(args.max_samples, len(cache)); negatives = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device); writer, _ = load_writer(args.writer, device); writer.requires_grad_(False); writer.eval()
    checkpoint = torch.load(args.reader, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = LearnableCanonicalHeadReader(model, metadata["selected_layers"], metadata["rank"], metadata["gate_init"], metadata["top_k"], 0.25).to(device); reader.load_state_dict(checkpoint["reader"]); reader.requires_grad_(False); reader.eval()
    rows, diagnostics = [], []
    for index in tqdm(range(count), desc=f"c3b_pair_ablation_seed{args.seed}"):
        payload, wrong = cache.load(index), cache.load(negatives[index]); row = payload["row"]
        correct_base, wrong_base = writer_memory(writer, payload, device, no_grad=True), writer_memory(writer, wrong, device, no_grad=True)
        baseline_trace = {}; baseline = generate(model, tokenizer, reader, row, correct_base, args.max_new_tokens, trace=baseline_trace)
        shuffled_baseline = generate(model, tokenizer, reader, row, wrong_base, args.max_new_tokens)
        for pair in range(8):
            for mode in MODES:
                if mode == "both":
                    for memory_condition, result in (("correct", baseline), ("hard_shuffled", shuffled_baseline)):
                        em, f1 = answer_scores(result["prediction"], row["answer"])
                        rows.append({"id": row["id"], "type": row["type"], "answer": row["answer"], "pair_index": pair, "pair": [2 * pair, 2 * pair + 1], "mode": mode,
                                     "memory_condition": memory_condition, "em": em, "f1": f1, "output": result,
                                     "source_id": wrong["row"]["id"] if memory_condition == "hard_shuffled" else row["id"], "prediction_changed_from_baseline": 0.0})
                    diagnostics.append({"id": row["id"], "pair_index": pair, "mode": mode, "layers": []})
                    continue
                changed_trace = {}
                correct = generate(model, tokenizer, reader, row, intervene_pair(correct_base, pair, mode), args.max_new_tokens, trace=changed_trace)
                shuffled = generate(model, tokenizer, reader, row, intervene_pair(wrong_base, pair, mode), args.max_new_tokens)
                for memory_condition, result in (("correct", correct), ("hard_shuffled", shuffled)):
                    em, f1 = answer_scores(result["prediction"], row["answer"])
                    rows.append({"id": row["id"], "type": row["type"], "answer": row["answer"], "pair_index": pair, "pair": [2 * pair, 2 * pair + 1], "mode": mode,
                                 "memory_condition": memory_condition, "em": em, "f1": f1, "output": result,
                                 "source_id": wrong["row"]["id"] if memory_condition == "hard_shuffled" else row["id"],
                                 "prediction_changed_from_baseline": float(normalize_answer(result["prediction"]) != normalize_answer(baseline["prediction"]))})
                diagnostics.append({"id": row["id"], "pair_index": pair, "mode": mode, "layers": trace_change(baseline_trace, changed_trace, pair, reader)})
    write_jsonl(output / "per_sample_generation.jsonl", rows); write_jsonl(output / "attention_readout_changes.jsonl", diagnostics)
    pairs = {}
    for pair in range(8):
        pairs[str(pair)] = {}
        for mode in MODES:
            correct, shuffled = summarize(rows, pair, mode, "correct"), summarize(rows, pair, mode, "hard_shuffled")
            pairs[str(pair)][mode] = {"correct": correct, "hard_shuffled": shuffled, "correct_shuffled_f1_gap": correct["f1"] - shuffled["f1"]}
    write_json(output / "SUCCESS.json", {"status": "complete", "experiment": "C3-B paired Canonical-head ablation", "seed": args.seed, "samples": count,
        "intervention_scope": "same Canonical head pair across all 16 memory layer groups", "modes": list(MODES), "pairs": pairs,
        "writer": args.writer, "reader": args.reader, "writer_frozen": True, "reader_frozen": True})


if __name__ == "__main__": main()
