import argparse
import gc
from pathlib import Path

import torch

from common import (ROOT, build_anchors, capture_decision, digest, encode, load_model, log,
                    read_json, save_json, save_tensor, select_rows, serialize,
                    spans_from_encoded, tokenizer)

FAMILIES = ("llama", "qwen", "gemma")


def run_root(mode):
    return ROOT / "runs" / mode


def signature(cfg, mode, dataset):
    return digest({"protocol": cfg["protocol"], "mode": mode, "dataset": dataset,
                   "samples": 2 if mode == "smoke" else cfg["samples_per_dataset"],
                   "regions": cfg["regions"]})


def manifest_path(mode, dataset):
    return run_root(mode) / dataset / "manifest.json"


def native_path(mode, dataset, family, sample_id):
    return run_root(mode) / dataset / "cache" / "native" / family / f"{sample_id}.pt"


def route_path(mode, dataset, sample_id):
    return run_root(mode) / dataset / "cache" / "routes" / f"{sample_id}.pt"


def prepare_manifest(cfg, mode, dataset):
    count = 2 if mode == "smoke" else cfg["samples_per_dataset"]
    rows, audit = select_rows(cfg, dataset, count)
    toks = {family: tokenizer(cfg["models"][family]) for family in FAMILIES}
    prepared, maxima = [], {family: 0 for family in FAMILIES}
    for row in rows:
        text = serialize(row)
        encoded = {family: encode(tok, text) for family, tok in toks.items()}
        for family in FAMILIES:
            maxima[family] = max(maxima[family], len(encoded[family]["body"]))
            if len(encoded[family]["body"]) > cfg["max_body_tokens"]:
                raise RuntimeError(f"Body too long: {row['id']} {family}")
        prepared.append({**row, **text, "encoded": encoded})
    sig = signature(cfg, mode, dataset)
    save_json(manifest_path(mode, dataset), {"signature": sig, "dataset": dataset, "rows": prepared})
    audit.update({"signature": sig, "max_body_tokens": maxima,
                  "option_counts": {str(n): sum(len(row["options"]) == n for row in prepared)
                                    for n in sorted({len(row["options"]) for row in prepared})}})
    save_json(run_root(mode) / dataset / "results" / "dataset_audit.json", audit)
    log(f"{dataset}: manifest prepared {len(prepared)} rows; max tokens={maxima}")


def load_manifest(cfg, mode, dataset):
    payload = read_json(manifest_path(mode, dataset))
    if payload["signature"] != signature(cfg, mode, dataset): raise RuntimeError("Manifest signature mismatch")
    return payload["rows"]


@torch.no_grad()
def cache_family(cfg, mode, dataset, family):
    rows = load_manifest(cfg, mode, dataset)
    model = load_model(cfg, family)
    try:
        for number, row in enumerate(rows, 1):
            path = native_path(mode, dataset, family, row["id"])
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("signature") == signature(cfg, mode, dataset):
                    continue
            fields = row["encoded"][family]
            key, value, importance, logits = capture_decision(
                model, fields["full"], len(fields["body"]), fields["choice_ids"])
            save_tensor(path, {"signature": signature(cfg, mode, dataset), "id": row["id"],
                "key": key.contiguous(), "value": value.contiguous(),
                "importance": importance, "full_choice_logits": logits})
            if number % 16 == 0 or number == len(rows):
                log(f"{dataset} native {family}: {number}/{len(rows)}")
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()


def build_routes(cfg, mode, dataset):
    rows = load_manifest(cfg, mode, dataset)
    sig = signature(cfg, mode, dataset)
    for number, row in enumerate(rows, 1):
        output = route_path(mode, dataset, row["id"])
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=True)
            if payload.get("signature") == sig: continue
        native = {family: torch.load(native_path(mode, dataset, family, row["id"]),
                                     map_location="cpu", weights_only=True) for family in FAMILIES}
        spans, option_spans = {}, {}
        for family in FAMILIES:
            spans[family] = spans_from_encoded(row["encoded"][family])
            allowed = set(row["encoded"][family]["option_token_indices"])
            option_spans[family] = [span for span in spans[family] if span.index in allowed]
        routes, metadata = {}, {}
        for router in FAMILIES:
            routes[router], metadata[router] = {}, {}
            for target in FAMILIES:
                anchors = build_anchors(row["body"], option_spans[router], option_spans[target],
                                        native[router]["importance"], cfg["regions"],
                                        *row["options_char_span"])
                indices = [anchor["target_index"] for anchor in anchors]
                routes[router][target] = {
                    "key": native[target]["key"][:, indices].contiguous(),
                    "value": native[target]["value"][:, indices].contiguous()}
                metadata[router][target] = {
                    "indices": indices, "unique_indices": len(set(indices)),
                    "source_indices": [anchor["source_index"] for anchor in anchors],
                    "anchor_chars": [anchor["anchor_char"] for anchor in anchors]}
        receiver = {}
        for family in FAMILIES:
            length = row["encoded"][family]["question_prefix_length"]
            receiver[family] = {"question_k": native[family]["key"][:, :length].contiguous(),
                                "question_v": native[family]["value"][:, :length].contiguous(),
                                "full_choice_logits": native[family]["full_choice_logits"]}
        save_tensor(output, {"signature": sig, "id": row["id"], "receiver": receiver,
                             "routes": routes, "metadata": metadata})
        if number % 16 == 0 or number == len(rows):
            log(f"{dataset} routes: {number}/{len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("--dataset", choices=("arc_challenge", "mmlu_pro", "all"), default="all")
    parser.add_argument("action", choices=("manifest", "native", "routes", "all"))
    args = parser.parse_args(); cfg = read_json(args.config)
    torch.manual_seed(cfg["seed"])
    datasets = ("arc_challenge", "mmlu_pro") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        if args.action in ("manifest", "all"): prepare_manifest(cfg, args.mode, dataset)
        if args.action in ("native", "all"):
            for family in FAMILIES: cache_family(cfg, args.mode, dataset, family)
        if args.action in ("routes", "all"): build_routes(cfg, args.mode, dataset)


if __name__ == "__main__":
    main()
