"""Deterministic synthetic RAG scenarios for regression tests.

A scenario is a small random corpus, a knowledge base holding part of it, a
"web" source returning first-sentence snippets of the whole corpus, a question,
and a scripted LLM whose reply is a fixed function of the prompt text. The
golden file ``golden_rag_v2.json`` holds the decisions the original tested
FGAO v2 implementation made on these scenarios.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Tuple

VOCAB = ("river delta harbor canal bridge tower castle market temple garden valley summit glacier forest "
         "meadow prairie desert canyon island lagoon reef volcano crater plateau basin estuary fjord cliff "
         "quarry orchard vineyard granary mill forge foundry kiln loom anvil chisel lathe compass sextant "
         "telescope lantern beacon").split()
NAMES = "Arvel Brisa Corin Dalen Esker Fenwick Galen Hollis Ivor Jessamy Kestrel Lorne Maren Norrin".split()


def _h(*parts: str) -> int:
    return int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12], 16)


def make_scenario(seed: int):
    rng = random.Random(seed)
    docs = []
    for i in range(rng.randint(12, 30)):
        sents = []
        for _ in range(rng.randint(2, 4)):
            words = rng.sample(VOCAB, rng.randint(5, 9))
            sents.append(f"The {words[0]} of {rng.choice(NAMES)} " + " ".join(words[1:]) + ".")
        docs.append({"id": f"d{i}", "title": rng.choice(NAMES) + " " + rng.choice(VOCAB).title(), "text": " ".join(sents)})
    kb_ids = {d["id"] for d in docs if rng.random() < rng.choice((0.3, 0.6, 0.9))}
    q_words = rng.sample(VOCAB, rng.randint(2, 5))
    question = f"What is the {' '.join(q_words)} of {rng.choice(NAMES)}?"
    budget = rng.choice((16.0, 16.0, 16.0, 8.0, 12.0, 24.0, 5.0))
    profile = {"p_abstain": rng.choice((0.1, 0.3, 0.6)), "p_yes": rng.choice((0.2, 0.6, 0.9)),
               "p_fabricate": rng.choice((0.0, 0.15, 0.4))}
    return {"seed": seed, "docs": docs, "kb_ids": sorted(kb_ids), "question": question, "budget": budget,
            "profile": profile}


def web_snippet(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text


def scripted_reply(prompt: str, profile: Dict[str, float]) -> str:
    """Deterministic stand-in for an LLM."""
    u = (_h("u", prompt) % 10_000) / 10_000.0
    if prompt.startswith("Answer the question using ONLY"):
        body = prompt.split("Passages:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
        passages = [re.sub(r"^\[\d+\]\s*(\([^)]*\)\s*)?", "", line) for line in body.splitlines() if line.strip()]
        if u < profile["p_abstain"] or not passages:
            return "unanswerable"
        if u > 1.0 - profile["p_fabricate"]:
            return "the " + VOCAB[_h("f", prompt) % len(VOCAB)] + " of nowhere"
        words = passages[_h("p", prompt) % len(passages)].rstrip(".").split()
        n = 1 + _h("n", prompt) % 3
        start = _h("s", prompt) % max(1, len(words) - n + 1)
        return " ".join(words[start:start + n])
    if prompt.startswith("Rate how useful"):
        n = prompt.count("\n[")
        return "\n".join(f"[{i + 1}] {_h('g', prompt, str(i)) % 3}" for i in range(n))
    if prompt.startswith("Rewrite the question"):
        q = prompt.rsplit("Question: ", 1)[1].split("\n", 1)[0]
        toks = [t for t in re.findall(r"[a-z]+", q.lower()) if len(t) > 3]
        keep = [t for i, t in enumerate(toks) if _h("r", prompt, str(i)) % 3]
        return " ".join(keep or toks)
    if prompt.startswith("Does the passage fully support"):
        return "yes" if u < profile["p_yes"] else "no"
    raise ValueError("unexpected prompt")


def build(sc):
    """(kb_documents, all_documents) as dicts."""
    kb = [d for d in sc["docs"] if d["id"] in set(sc["kb_ids"])]
    return kb, sc["docs"]
