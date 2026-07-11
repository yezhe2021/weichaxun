import argparse
import json
import random
from pathlib import Path


SCHEMAS = [
    {
        "name": "employment_location",
        "types": ("Person", "Organization", "Place"),
        "r1": ["{x} works for {z}.", "{z} employs {x}.", "{x} is employed by {z}."],
        "r2": ["{z} is based in {y}.", "{z} has its headquarters in {y}.", "The home office of {z} is in {y}."],
        "q": ["Where is the organization employing {x} based?", "In which place is {x}'s employer headquartered?"],
    },
    {
        "name": "education_country",
        "types": ("Student", "School", "Country"),
        "r1": ["{x} studied at {z}.", "{x} attended {z}.", "{z} educated {x}."],
        "r2": ["{z} is located in {y}.", "{z} is an institution in {y}.", "The country containing {z} is {y}."],
        "q": ["In which country is the school attended by {x}?", "Where is {x}'s school located?"],
    },
    {
        "name": "actor_director",
        "types": ("Actor", "Film", "Director"),
        "r1": ["{x} appeared in {z}.", "{x} acted in {z}.", "The cast of {z} included {x}."],
        "r2": ["{z} was directed by {y}.", "{y} directed {z}.", "The director of {z} was {y}."],
        "q": ["Who directed the film featuring {x}?", "Name the director of the film in which {x} appeared."],
    },
    {
        "name": "book_nationality",
        "types": ("Book", "Author", "Nationality"),
        "r1": ["{x} was written by {z}.", "{z} authored {x}.", "The author of {x} is {z}."],
        "r2": ["{z} has nationality {y}.", "{z} is {y}.", "The nationality of {z} is {y}."],
        "q": ["What is the nationality of the author of {x}?", "Which nationality does the writer of {x} have?"],
    },
    {
        "name": "research_venue",
        "types": ("Researcher", "Paper", "Venue"),
        "r1": ["{x} wrote {z}.", "{z} was authored by {x}.", "{x} is the researcher behind {z}."],
        "r2": ["{z} was published at {y}.", "{y} published {z}.", "The venue for {z} was {y}."],
        "q": ["At which venue was the paper by {x} published?", "Where was {x}'s paper published?"],
    },
    {
        "name": "album_producer",
        "types": ("Singer", "Album", "Producer"),
        "r1": ["{x} recorded {z}.", "{z} was recorded by {x}.", "{x} is the singer on {z}."],
        "r2": ["{z} was produced by {y}.", "{y} produced {z}.", "The producer of {z} was {y}."],
        "q": ["Who produced the album recorded by {x}?", "Name the producer of {x}'s album."],
    },
    {
        "name": "product_country",
        "types": ("Product", "Maker", "Country"),
        "r1": ["{x} is made by {z}.", "{z} manufactures {x}.", "The maker of {x} is {z}."],
        "r2": ["{z} operates from {y}.", "{z} is a manufacturer in {y}.", "The home country of {z} is {y}."],
        "q": ["In which country is the maker of {x} based?", "What is the home country of {x}'s manufacturer?"],
    },
    {
        "name": "event_city",
        "types": ("Event", "Venue", "City"),
        "r1": ["{x} took place at {z}.", "{z} hosted {x}.", "The venue for {x} was {z}."],
        "r2": ["{z} is in {y}.", "{y} contains {z}.", "The city of {z} is {y}."],
        "q": ["In which city was {x} held?", "What city contains the venue that hosted {x}?"],
    },
    {
        "name": "object_room",
        "types": ("Object", "Container", "Room"),
        "r1": ["{x} is stored in {z}.", "{z} contains {x}.", "The container holding {x} is {z}."],
        "r2": ["{z} is inside {y}.", "{y} contains {z}.", "The room holding {z} is {y}."],
        "q": ["Which room contains the container holding {x}?", "Where is {x}'s container located?"],
    },
    {
        "name": "species_climate",
        "types": ("Species", "Habitat", "Climate"),
        "r1": ["{x} lives in {z}.", "{z} is the habitat of {x}.", "{x} inhabits {z}."],
        "r2": ["{z} has a {y} climate.", "The climate of {z} is {y}.", "{y} conditions characterize {z}."],
        "q": ["What climate characterizes the habitat of {x}?", "Which climate does the habitat of {x} have?"],
    },
    {
        "name": "route_destination",
        "types": ("Vehicle", "Route", "Destination"),
        "r1": ["{x} follows {z}.", "{z} is assigned to {x}.", "The route used by {x} is {z}."],
        "r2": ["{z} terminates at {y}.", "The final stop of {z} is {y}.", "{y} is the destination of {z}."],
        "q": ["What is the destination of the route followed by {x}?", "Where does the route used by {x} terminate?"],
    },
    {
        "name": "team_leader",
        "types": ("Member", "Team", "Leader"),
        "r1": ["{x} belongs to {z}.", "{z} includes {x}.", "{x} is a member of {z}."],
        "r2": ["{z} is led by {y}.", "{y} leads {z}.", "The leader of {z} is {y}."],
        "q": ["Who leads the team containing {x}?", "Name the leader of {x}'s team."],
    },
]


def entity(kind, number):
    return f"{kind}-{number:05d}"


def make_pair(schema, pair_number, rng, split, candidates, a_distractors, template_ood):
    x_type, z_type, y_type = schema["types"]
    base = pair_number * 100
    target_x = entity(x_type, base + 1)
    target_z = entity(z_type, base + 2)
    target_y = entity(y_type, base + 3)
    counterfactual_y = entity(y_type, base + 4)

    bridge_entities = [target_z] + [entity(z_type, base + 10 + i) for i in range(candidates - 1)]
    answer_entities = [target_y] + [entity(y_type, base + 30 + i) for i in range(candidates - 1)]
    rng.shuffle(bridge_entities)
    target_index = bridge_entities.index(target_z)
    answer_entities[target_index], answer_entities[0] = answer_entities[0], answer_entities[target_index]

    r1_index = 2 if template_ood else rng.randrange(0, 2)
    r2_index = 2 if template_ood else rng.randrange(0, 2)
    q_index = 1 if template_ood else 0
    facts_a = [schema["r1"][r1_index].format(x=target_x, z=target_z)]
    for index in range(a_distractors):
        distractor_x = entity(x_type, base + 50 + index)
        distractor_z = entity(z_type, base + 60 + index)
        facts_a.append(schema["r1"][r1_index].format(x=distractor_x, z=distractor_z))
    rng.shuffle(facts_a)

    facts_b = [
        schema["r2"][r2_index].format(z=bridge, y=answer)
        for bridge, answer in zip(bridge_entities, answer_entities)
    ]
    question = schema["q"][q_index].format(x=target_x)
    pair_id = f"{split}-{schema['name']}-{pair_number:07d}"

    rows = []
    for variant, answer in (("base", target_y), ("counterfactual", counterfactual_y)):
        variant_facts_b = list(facts_b)
        if variant == "counterfactual":
            variant_facts_b[target_index] = schema["r2"][r2_index].format(z=target_z, y=counterfactual_y)
        rows.append(
            {
                "id": f"{pair_id}-{variant}",
                "pair_id": pair_id,
                "variant": variant,
                "split": split,
                "schema": schema["name"],
                "question": question,
                "evidence_a": " ".join(facts_a),
                "evidence_b": " ".join(variant_facts_b),
                "answer": answer,
                "base_answer": target_y,
                "counterfactual_answer": counterfactual_y,
                "bridge": target_z,
                "candidate_bridges": bridge_entities,
                "candidate_answers": [
                    counterfactual_y if variant == "counterfactual" and i == target_index else value
                    for i, value in enumerate(answer_entities)
                ],
                "target_candidate_index": target_index,
                "template_ood": template_ood,
            }
        )
    return rows


def write_split(path, split, pairs, seed, offset, candidates, a_distractors, template_ood):
    rng = random.Random(seed + offset)
    rows = []
    for index in range(pairs):
        schema = SCHEMAS[index % len(SCHEMAS)]
        pair_number = offset * 10_000 + index
        rows.extend(make_pair(schema, pair_number, rng, split, candidates, a_distractors, template_ood))
    rng.shuffle(rows)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-schema causal two-hop data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--valid-pairs", type=int, default=256)
    parser.add_argument("--test-pairs", type=int, default=512)
    parser.add_argument("--b-candidates", type=int, default=4)
    parser.add_argument("--a-distractors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": write_split(output / "train.jsonl", "train", args.train_pairs, args.seed, 1, args.b_candidates, args.a_distractors, False),
        "valid": write_split(output / "valid.jsonl", "valid", args.valid_pairs, args.seed, 2, args.b_candidates, args.a_distractors, False),
        "test": write_split(output / "test.jsonl", "test", args.test_pairs, args.seed, 3, args.b_candidates, args.a_distractors, True),
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "schemas": [schema["name"] for schema in SCHEMAS], "rows": counts}, handle, indent=2)


if __name__ == "__main__":
    main()
