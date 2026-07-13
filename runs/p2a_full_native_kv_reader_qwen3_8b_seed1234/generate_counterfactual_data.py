import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from p2a_common import write_jsonl


FIRST_NAMES = [
    "Alice", "Bruno", "Celine", "Daria", "Elias", "Farah", "Gavin", "Hana", "Iris", "Jonas",
    "Kara", "Leon", "Maya", "Nadia", "Omar", "Petra", "Quinn", "Rina", "Soren", "Talia",
    "Uma", "Vera", "Wade", "Xenia", "Yara", "Zane", "Amira", "Basil", "Clara", "Damon",
    "Elena", "Felix", "Greta", "Hugo", "Ines", "Jules", "Keira", "Luca", "Mina", "Nolan",
]

LAST_NAMES = [
    "Voss", "Mercer", "Rowan", "Klein", "Navarro", "Bennett", "Sato", "Dawson", "Ibarra", "Novak",
    "Fischer", "Moreau", "Silva", "Petrov", "Khan", "Larsen", "Costa", "Reed", "Weiss", "Chen",
]

COMPANY_LEFT = [
    "Amber", "Birch", "Cobalt", "Delta", "Elm", "Frost", "Granite", "Harbor", "Indigo", "Juniper",
    "Keystone", "Lumen", "Maple", "Nimbus", "Orchid", "Pioneer", "Quartz", "Redwood", "Summit", "Thistle",
]

COMPANY_RIGHT = [
    "Analytics", "Dynamics", "Foundry", "Holdings", "Industries", "Laboratories", "Networks", "Partners",
    "Research", "Robotics", "Systems", "Technologies", "Ventures", "Works",
]

LOCATIONS = [
    "Athens", "Berlin", "Boston", "Dublin", "Geneva", "Lisbon", "London", "Madrid", "Milan", "Oslo",
    "Paris", "Prague", "Rome", "Seoul", "Sydney", "Tokyo", "Vienna", "Warsaw", "Zurich", "Denver",
    "Helsinki", "Jakarta", "Kingston", "Lima", "Manila", "Nairobi", "Ottawa", "Riga", "Sofia", "Tallinn",
    "Valencia", "Brisbane", "Calgary", "Florence", "Hamburg", "Montreal", "Naples", "Portland", "Seattle", "Toronto",
]


def make_names():
    return [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]


def make_companies():
    return [f"{left} {right}" for left in COMPANY_LEFT for right in COMPANY_RIGHT]


def compatible_location_groups(tokenizer):
    groups = {}
    for location in LOCATIONS:
        length = len(tokenizer(location, add_special_tokens=False).input_ids)
        groups.setdefault(length, []).append(location)
    return [values for values in groups.values() if len(values) >= 6]


def make_pair(split, pair_index, rng, tokenizer, names, companies, location_groups):
    people = rng.sample(names, 4)
    organizations = rng.sample(companies, 4)
    locations = rng.sample(rng.choice(location_groups), 6)
    answer, counterfactual = locations[0], locations[1]
    distractor_locations = locations[2:5]

    facts_a = [f"{person} works for {company}." for person, company in zip(people, organizations)]
    target_b = f"{organizations[0]} is located in {answer}."
    counterfactual_b = f"{organizations[0]} is located in {counterfactual}."
    facts_b = [target_b] + [
        f"{company} is located in {location}."
        for company, location in zip(organizations[1:], distractor_locations)
    ]
    target_position = 0
    permutation = list(range(len(facts_b)))
    rng.shuffle(permutation)
    target_position = permutation.index(0)
    facts_a_ordered = list(facts_a)
    rng.shuffle(facts_a_ordered)
    base_facts_b = [facts_b[index] for index in permutation]
    cf_facts_b = list(base_facts_b)
    cf_facts_b[target_position] = counterfactual_b

    pair_id = f"{split}-pair-{pair_index:05d}"
    common = {
        "pair_id": pair_id,
        "split": split,
        "question": f"In which city is the employer of {people[0]} located?",
        "evidence_a": " ".join(facts_a_ordered),
        "target_person": people[0],
        "target_organization": organizations[0],
        "target_relation_index": target_position,
    }
    base_candidates = [answer, *distractor_locations]
    cf_candidates = [counterfactual, *distractor_locations]
    return [
        {
            **common,
            "id": f"{pair_id}-base",
            "variant": "base",
            "evidence_b": " ".join(base_facts_b),
            "answer": answer,
            "counterpart_answer": counterfactual,
            "candidate_answers": base_candidates,
        },
        {
            **common,
            "id": f"{pair_id}-counterfactual",
            "variant": "counterfactual",
            "evidence_b": " ".join(cf_facts_b),
            "answer": counterfactual,
            "counterpart_answer": answer,
            "candidate_answers": cf_candidates,
        },
    ]


def build_split(name, pairs, offset, seed, tokenizer):
    rng = random.Random(f"{seed}:{name}")
    names = make_names()
    companies = make_companies()
    groups = compatible_location_groups(tokenizer)
    rows = []
    for index in range(pairs):
        rows.extend(make_pair(name, offset + index, rng, tokenizer, names, companies, groups))
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate strict paired counterfactual P2-A data")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--test-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    train = build_split("train", args.train_pairs, 0, args.seed, tokenizer)
    test = build_split("test", args.test_pairs, 10000, args.seed, tokenizer)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "train_rows": len(train),
                "test_rows": len(test),
                "counterfactual_constraints": {
                    "same_question": True,
                    "same_evidence_a": True,
                    "same_bridge": True,
                    "same_target_relation_position": True,
                    "equal_answer_token_length": True,
                    "answer_absent_from_question": True,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
