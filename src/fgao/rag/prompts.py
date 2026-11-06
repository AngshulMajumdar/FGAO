"""Prompts and parsers of the evaluated v2 RAG operations (unchanged wording)."""
from __future__ import annotations

import re
from typing import List, Sequence

from .documents import Document

SYSTEM_PROMPT = "You are a careful, concise assistant."


def _label(d: Document) -> str:
    return f"({d.title}) {d.text}" if d.title else d.text


def format_passages(docs: Sequence[Document]) -> str:
    return "\n".join(f"[{i + 1}] {_label(d)}" for i, d in enumerate(docs))


def generate_prompt(question: str, docs: Sequence[Document]) -> str:
    return (
        "Answer the question using ONLY the passages below. Reply with the shortest exact answer span "
        "copied from the passages. If the passages do not contain the answer, reply exactly: unanswerable\n\n"
        f"Passages:\n{format_passages(docs)}\n\nQuestion: {question}\nAnswer:"
    )


def grade_prompt(question: str, docs: Sequence[Document]) -> str:
    return (
        "Rate how useful each passage is for answering the question.\n"
        "2 = it contains the answer, 1 = it is on the topic but does not clearly contain the answer, 0 = irrelevant.\n"
        "Reply with exactly one line per passage in the form '[i] score' and nothing else.\n\n"
        f"Question: {question}\n\nPassages:\n{format_passages(docs)}"
    )


def rewrite_prompt(question: str) -> str:
    return ("Rewrite the question as a short keyword search query (at most 8 words) that would retrieve a passage "
            f"containing the answer. Reply with the query only.\n\nQuestion: {question}\nQuery:")


def verify_prompt(question: str, passage: Document, answer: str) -> str:
    return (
        "Does the passage fully support the given answer to the question? Reply exactly 'yes' or 'no'.\n\n"
        f"Passage: {_label(passage)}\n\nQuestion: {question}\nAnswer: {answer}\nSupported:"
    )


# ------------------------------------------------------------------ parsers
_ABSTAIN = {"unanswerable", "unknown", "no answer", "none", "n/a", "cannot be determined", "not stated"}


def parse_answer(text: str) -> str:
    """First line of the reply, stripped; '' for an abstention."""
    t = (text or "").strip()
    m = re.search(r"answer\s*:\s*(.+)", t, re.I)
    if m:
        t = m.group(1)
    t = t.strip().splitlines()[0].strip() if t.strip() else ""
    t = t.strip(" \"'`*.")
    return "" if t.lower().rstrip(".") in _ABSTAIN else t


def parse_grades(text: str, n: int) -> List[int]:
    out = [0] * n
    for i, s in re.findall(r"\[(\d+)\]\s*[:=-]?\s*([012])", text or ""):
        k = int(i) - 1
        if 0 <= k < n:
            out[k] = int(s)
    return out


def parse_yes(text: str) -> bool:
    return (text or "").strip().lower().startswith("yes")


def parse_query(text: str) -> str:
    t = (text or "").strip()
    return t.splitlines()[0] if t else ""
