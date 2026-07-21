import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from p3d_common import MultiLayerEvidenceReader, answer_scores, extract_prediction, generate, load_receiver, memory_to, read_json, resize_memory, write_json, write_jsonl, zero_memory
from p3d2_common import canonical_cache, hard_negative_mapping, install_oracle_forward, load_span_probe, memory_with_oracle, span_teacher


def summarize(records):
    grouped = defaultdict(list)
    for row in records: grouped[(row["mode"], row["condition"])].append(row)
    output = []
    for (mode, condition), rows in sorted(grouped.items()):
        item = {"mode": mode, "condition": condition, "n": len(rows), "em": sum(row["em"] for row in rows) / len(rows), "f1": sum(row["f1"] for row in rows) / len(rows), "source_em": sum(row["source_em"] for row in rows) / len(rows)}
        for kind in ("bridge", "comparison"):
            subset = [row for row in rows if row["question_type"] == kind]
            if subset: item[f"{kind}_f1"] = sum(row["f1"] for row in subset) / len(subset)
        output.append(item)
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--reader", required=True); parser.add_argument("--out", required=True); parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--top-groups", type=int, default=4); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); protocol = read_json(args.protocol); cache = canonical_cache(protocol, "test")
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.reader, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = MultiLayerEvidenceReader(model, metadata["groups"], metadata["memory_dim"], metadata["rank"], metadata["adapter_rank"], active_layers=metadata["active_layers"]).to(device)
    reader.load_state_dict(checkpoint["reader"]); reader.eval(); install_oracle_forward(reader)
    probe, probe_path = load_span_probe(protocol, device); negative = hard_negative_mapping(cache)
    limit = min(args.max_samples, len(cache)); records = []
    for index in tqdm(range(limit), desc="p3d2_oracle"):
        current, wrong = cache.load(index), cache.load(negative[index])
        current_ground = span_teacher(probe, current, device, args.top_groups)
        wrong_for_query = dict(wrong); wrong_for_query["question_state"] = current["question_state"]
        wrong_ground = span_teacher(probe, wrong_for_query, device, args.top_groups)
        for mode in ("ordinary", "oracle_token", "oracle_token_layer"):
            for condition in ("correct", "shuffled", "zero", "reader_off"):
                source = wrong if condition == "shuffled" else current
                grounding = wrong_ground if condition == "shuffled" else current_ground
                token_oracle = mode != "ordinary"
                group_mask = grounding["group_mask"] if mode == "oracle_token_layer" else None
                memory = memory_with_oracle(source, device, token_oracle, group_mask)
                if condition == "zero": memory = zero_memory(memory)
                enabled = condition != "reader_off"
                result = generate(model, tokenizer, reader, current["row"], memory, args.max_new_tokens, enabled)
                prediction, parse_status = extract_prediction(result["text"])
                em, f1 = answer_scores(prediction, current["row"]["answer"]); source_em, source_f1 = answer_scores(prediction, source["row"]["answer"])
                records.append({"sample_index": index, "sample_id": current["row"].get("id", str(index)), "mode": mode, "condition": condition, "question_type": current["row"].get("type", "unknown"), "prediction": prediction, "raw_generation": result["text"], "parse_status": parse_status, "gold_answer": current["row"]["answer"], "source_answer": source["row"]["answer"], "em": em, "f1": f1, "source_em": source_em, "source_f1": source_f1, "eos_reached": result["eos_reached"], "oracle_groups": grounding["group_mask"].nonzero(as_tuple=False).flatten().cpu().tolist() if mode == "oracle_token_layer" else []})
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records)
    summary = summarize(records)
    gaps = {}
    for mode in ("ordinary", "oracle_token", "oracle_token_layer"):
        correct = next(row for row in summary if row["mode"] == mode and row["condition"] == "correct")
        shuffled = next(row for row in summary if row["mode"] == mode and row["condition"] == "shuffled")
        gaps[mode] = {"correct_minus_shuffled_f1": correct["f1"] - shuffled["f1"], "bridge_f1": correct.get("bridge_f1")}
    write_json(output / "SUCCESS.json", {"status": "complete", "samples": limit, "conditions": summary, "causal_gaps": gaps, "reader_checkpoint": args.reader, "span_probe": str(probe_path), "writer_frozen": True, "receiver_frozen": True})


if __name__ == "__main__": main()
