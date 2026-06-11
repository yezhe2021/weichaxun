import argparse
import json
from pathlib import Path


def flatten_context(context, max_chars):
    chunks = []
    for title, sentences in context:
        text = " ".join(s.strip() for s in sentences if s.strip())
        if text:
            chunks.append(f"[{title}] {text}")
    joined = "\n".join(chunks)
    return joined[:max_chars]


def convert(src, dst, limit, max_context_chars):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(src, encoding="utf-8"))
    written = 0
    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            context = flatten_context(row.get("context", []), max_context_chars)
            question = row.get("question", "")
            answer = row.get("answer", "")
            if not context or not question or not answer:
                continue
            out = {
                "id": row.get("_id", ""),
                "context": context,
                "question": question,
                "answer": answer,
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "supporting_facts": row.get("supporting_facts", []),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    print(f"wrote {written} examples to {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hotpot-root", default="/home/yezhe/数据集/HotpotQA")
    p.add_argument("--out-dir", default="/home/yezhe/数据集/HotpotQA/processed")
    p.add_argument("--train-limit", type=int, default=2000)
    p.add_argument("--dev-limit", type=int, default=500)
    p.add_argument("--max-context-chars", type=int, default=6000)
    args = p.parse_args()

    root = Path(args.hotpot_root)
    out = Path(args.out_dir)
    convert(root / "raw/hotpot_train_v1.1.json", out / "hotpot_train_context_qa.jsonl", args.train_limit, args.max_context_chars)
    convert(root / "raw/hotpot_dev_distractor_v1.json", out / "hotpot_dev_context_qa.jsonl", args.dev_limit, args.max_context_chars)


if __name__ == "__main__":
    main()
