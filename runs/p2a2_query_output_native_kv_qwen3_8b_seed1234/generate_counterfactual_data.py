import argparse
import json
import random
from collections import Counter
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


class BalancedCycle:
    def __init__(self, values, rng):
        self.values = list(values)
        self.rng = rng
        self.buffer = []

    def take(self):
        if not self.buffer:
            self.buffer = list(self.values)
            self.rng.shuffle(self.buffer)
        return self.buffer.pop()


def make_names():
    return [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]


def make_companies():
    return [f"{left} {right}" for left in COMPANY_LEFT for right in COMPANY_RIGHT]


def location_groups(tokenizer):
    groups = {}
    for location in LOCATIONS:
        token_count = len(tokenizer(location, add_special_tokens=False).input_ids)
        groups.setdefault(token_count, []).append(location)
    groups = {length: values for length, values in groups.items() if len(values) >= 4}
    usable = [location for values in groups.values() for location in values]
    if len(usable) < 20:
        raise ValueError("Too few locations have compatible answer token lengths")
    return groups, usable


def balanced_counterfactual(base, groups, counts, rng):
    token_group = next(values for values in groups.values() if base in values)
    candidates = [value for value in token_group if value != base]
    minimum = min(counts[value] for value in candidates)
    choices = [value for value in candidates if counts[value] == minimum]
    selected = rng.choice(choices)
    counts[selected] += 1
    return selected


def take_distinct(cycle, count, excluded):
    selected = []
    attempts = 0
    while len(selected) < count:
        value = cycle.take()
        attempts += 1
        if value not in excluded and value not in selected:
            selected.append(value)
        if attempts > 10000:
            raise RuntimeError("Unable to draw distinct balanced values")
    return selected


def frequency_summary(rows):
    fields = {
        "target_person": Counter(row["target_person"] for row in rows if row["variant"] == "base"),
        "target_organization": Counter(row["target_organization"] for row in rows if row["variant"] == "base"),
        "base_answer": Counter(row["answer"] for row in rows if row["variant"] == "base"),
        "counterfactual_answer": Counter(row["answer"] for row in rows if row["variant"] == "counterfactual"),
    }
    summary = {}
    for name, counts in fields.items():
        values = list(counts.values())
        summary[name] = {
            "unique": len(counts),
            "minimum": min(values),
            "maximum": max(values),
            "max_minus_min": max(values) - min(values),
            "counts": dict(sorted(counts.items())),
        }
    return summary


def build_split(name, pairs, seed, tokenizer):
    groups, usable_locations = location_groups(tokenizer)
    people = make_names()
    companies = make_companies()
    person_cycle = BalancedCycle(people, random.Random(f"{seed}:{name}:target-person"))
    company_cycle = BalancedCycle(companies, random.Random(f"{seed}:{name}:target-company"))
    base_city_cycle = BalancedCycle(usable_locations, random.Random(f"{seed}:{name}:base-city"))
    distractor_people = BalancedCycle(people, random.Random(f"{seed}:{name}:distractor-person"))
    distractor_companies = BalancedCycle(companies, random.Random(f"{seed}:{name}:distractor-company"))
    pair_rng = random.Random(f"{seed}:{name}:pair-layout")
    cf_rng = random.Random(f"{seed}:{name}:counterfactual-city")
    cf_counts = Counter({location: 0 for location in usable_locations})
    rows = []

    for pair_index in range(pairs):
        target_person = person_cycle.take()
        target_company = company_cycle.take()
        answer = base_city_cycle.take()
        counterfactual = balanced_counterfactual(answer, groups, cf_counts, cf_rng)
        people_for_pair = [target_person] + take_distinct(distractor_people, 3, {target_person})
        companies_for_pair = [target_company] + take_distinct(distractor_companies, 3, {target_company})
        distractor_locations = pair_rng.sample(
            [location for location in usable_locations if location not in {answer, counterfactual}], 3
        )

        facts_a = [
            f"{person} works for {company}."
            for person, company in zip(people_for_pair, companies_for_pair)
        ]
        pair_rng.shuffle(facts_a)
        facts_b = [f"{target_company} is located in {answer}."] + [
            f"{company} is located in {location}."
            for company, location in zip(companies_for_pair[1:], distractor_locations)
        ]
        permutation = list(range(4))
        pair_rng.shuffle(permutation)
        target_position = permutation.index(0)
        base_facts_b = [facts_b[index] for index in permutation]
        cf_facts_b = list(base_facts_b)
        cf_facts_b[target_position] = f"{target_company} is located in {counterfactual}."

        pair_id = f"{name}-pair-{pair_index:05d}"
        common = {
            "pair_id": pair_id,
            "split": name,
            "question": f"In which city is the employer of {target_person} located?",
            "evidence_a": " ".join(facts_a),
            "target_person": target_person,
            "target_organization": target_company,
            "target_relation_index": target_position,
        }
        rows.extend(
            [
                {
                    **common,
                    "id": f"{pair_id}-base",
                    "variant": "base",
                    "evidence_b": " ".join(base_facts_b),
                    "answer": answer,
                    "counterpart_answer": counterfactual,
                    "candidate_answers": [answer, *distractor_locations],
                },
                {
                    **common,
                    "id": f"{pair_id}-counterfactual",
                    "variant": "counterfactual",
                    "evidence_b": " ".join(cf_facts_b),
                    "answer": counterfactual,
                    "counterpart_answer": answer,
                    "candidate_answers": [counterfactual, *distractor_locations],
                },
            ]
        )

    summary = frequency_summary(rows)
    for field in ("target_person", "target_organization", "base_answer", "counterfactual_answer"):
        if summary[field]["max_minus_min"] > 1:
            raise AssertionError(f"Unbalanced {field}: {summary[field]}")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Generate balanced strict counterfactual P2-A2 data")
    parser.add_argument("--model", default="/home/yezhe/all_models/models/Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=512)
    parser.add_argument("--test-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    train, train_stats = build_split("train", args.train_pairs, args.seed, tokenizer)
    test, test_stats = build_split("test", args.test_pairs, args.seed + 1, tokenizer)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "train_rows": len(train),
                "test_rows": len(test),
                "constraints": {
                    "strict_counterfactual_pairs": True,
                    "equal_answer_token_length": True,
                    "independent_field_random_streams": True,
                    "maximum_role_frequency_gap": 1,
                },
                "train_frequency": train_stats,
                "test_frequency": test_stats,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
