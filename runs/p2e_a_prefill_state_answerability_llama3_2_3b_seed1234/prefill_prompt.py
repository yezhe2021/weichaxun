SYSTEM = (
    "Answer by joining the exact person in Evidence A to an employer, then joining that exact employer "
    "in Evidence B to a city. Ignore distractors. All required facts are present. Finish with FINAL: <city>."
)

EXAMPLE = (
    "EXAMPLE\n"
    "QUESTION\nIn which city is the employer of Mina Cole located?\n\n"
    "EVIDENCE A\nTheo Park works for Cedar Labs. Mina Cole works for Aurora Systems.\n\n"
    "EVIDENCE B\nCedar Labs is located in Rome. Aurora Systems is located in Oslo.\n\n"
    "REASONING\nMina Cole -> Aurora Systems -> Oslo\nFINAL: Oslo\n\n"
    "NOW SOLVE\n"
)


def condition_evidence(row, condition):
    if condition == "correct":
        return row["evidence_a"], row["evidence_b"]
    if condition == "question_only":
        return "[REMOVED]", "[REMOVED]"
    if condition == "a_only":
        return row["evidence_a"], "[REMOVED]"
    if condition == "b_only":
        return "[REMOVED]", row["evidence_b"]
    if condition == "answer_masked":
        masked = row["evidence_b"].replace(row["answer"], "[MASKED]", 1)
        if masked == row["evidence_b"]:
            raise ValueError(f"Answer {row['answer']!r} is absent from Evidence B")
        return row["evidence_a"], masked
    raise ValueError(f"Unknown condition: {condition}")


def evidence_block(row, condition="correct"):
    evidence_a, evidence_b = condition_evidence(row, condition)
    return (
        f"QUESTION\n{row['question']}\n\n"
        f"EVIDENCE A\n{evidence_a}\n\n"
        f"EVIDENCE B\n{evidence_b}"
    )


def render_prompt(tokenizer, row, condition="correct"):
    user = EXAMPLE + evidence_block(row, condition)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYSTEM}\n\n{user}\n\n"
    evidence_spans = None
    if condition == "correct":
        marker = prompt.rfind("NOW SOLVE")
        a_start = prompt.find(row["evidence_a"], marker)
        b_start = prompt.find(row["evidence_b"], a_start + len(row["evidence_a"]))
        if marker < 0 or a_start < 0 or b_start < 0:
            raise ValueError(f"Could not locate evidence spans for {row['id']}")
        evidence_spans = (
            (a_start, a_start + len(row["evidence_a"])),
            (b_start, b_start + len(row["evidence_b"])),
        )
    return prompt, evidence_spans


def choose_fixed_summary_token(tokenizer):
    special_ids = set(tokenizer.all_special_ids)
    for text in (" summary", " memory", " slot", " note", "\n"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in special_ids:
            return ids[0], text
    raise RuntimeError("Could not find a neutral one-token summary slot marker")


__all__ = [
    "EXAMPLE",
    "SYSTEM",
    "choose_fixed_summary_token",
    "condition_evidence",
    "evidence_block",
    "render_prompt",
]
