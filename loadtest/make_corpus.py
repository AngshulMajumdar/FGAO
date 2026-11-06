"""Synthetic corpus + question set for load testing.

    python make_corpus.py --entities 3000 --filler 20000 --out data/

Writes kb.jsonl (70% of the facts + filler), web.jsonl (every fact as a short
snippet + filler) and questions.jsonl ({"question", "expected"}; expected is
null for entities that do not exist, where the right output is an abstention).
"""
import argparse
import json
import os
import random

SYL = "ka lo mi ra ve tu sen dor bel fin gar hal ion jur kem lan mor nev ost pra quin rul sav tor ulm vex wyn zar".split()
WORDS = ("river valley harbor market temple garden summit forest meadow desert canyon island lagoon reef plateau "
         "basin cliff orchard vineyard mill forge kiln loom anvil lantern beacon festival railway museum library "
         "bridge tower castle canal granary quarry").split()


def name(rng, used):
    while True:
        n = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize()
        if n not in used:
            used.add(n)
            return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=3000)
    ap.add_argument("--filler", type=int, default=20000)
    ap.add_argument("--questions", type=int, default=4000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    used = set()
    facts = [(name(rng, used), name(rng, used)) for _ in range(a.entities)]  # (entity, capital)
    ghosts = [name(rng, used) for _ in range(max(1, a.entities // 10))]
    in_kb = {e for e, _ in facts if rng.random() < 0.7}
    os.makedirs(a.out, exist_ok=True)

    def filler(i):
        w = rng.sample(WORDS, 8)
        return {"id": f"f{i}", "title": w[0].title(), "text": f"The {w[0]} near the {w[1]} has a {w[2]} and a {w[3]}. "
                f"Visitors come for the {w[4]}, the {w[5]} and the {w[6]} by the {w[7]}."}

    with open(os.path.join(a.out, "kb.jsonl"), "w") as fh:
        for e, c in facts:
            if e in in_kb:
                w = rng.sample(WORDS, 3)
                fh.write(json.dumps({"id": f"kb-{e}", "title": e, "text": f"{e} is a country known for its {w[0]} and "
                                     f"{w[1]}. {c} is the capital of {e}. Its {w[2]} is famous."}) + "\n")
        for i in range(a.filler):
            fh.write(json.dumps(filler(i)) + "\n")
    with open(os.path.join(a.out, "web.jsonl"), "w") as fh:
        for e, c in facts:
            fh.write(json.dumps({"id": f"web-{e}", "title": f"{e} (web)", "text": f"{c} is the capital of {e}."}) + "\n")
        for i in range(a.filler // 4):
            fh.write(json.dumps(filler(a.filler + i)) + "\n")
    with open(os.path.join(a.out, "questions.jsonl"), "w") as fh:
        for _ in range(a.questions):
            if rng.random() < 0.1:
                fh.write(json.dumps({"question": f"What is the capital of {rng.choice(ghosts)}?", "expected": None}) + "\n")
            else:
                e, c = rng.choice(facts)
                fh.write(json.dumps({"question": f"What is the capital of {e}?", "expected": c,
                                     "source": "kb" if e in in_kb else "web"}) + "\n")
    print(f"{len(facts)} entities ({len(in_kb)} in KB), {a.filler} filler paragraphs, {a.questions} questions -> {a.out}")


if __name__ == "__main__":
    main()
