"""Documents and text utilities used by the RAG pipeline."""
from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Document:
    """A retrieved passage."""

    text: str
    title: str = ""
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha1((self.title + "\x00" + self.text).encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text, "metadata": dict(self.metadata)}


DocLike = Union[Document, str, Dict[str, Any]]


def as_document(x: DocLike) -> Document:
    if isinstance(x, Document):
        return x
    if isinstance(x, str):
        return Document(text=x)
    if isinstance(x, dict):
        meta = {k: v for k, v in x.items() if k not in ("text", "title", "id", "page_content")}
        return Document(text=x.get("text", x.get("page_content", "")), title=x.get("title", ""),
                        id=str(x.get("id", "")), metadata=meta)
    # LangChain-style document
    text = getattr(x, "page_content", None)
    if text is not None:
        meta = dict(getattr(x, "metadata", {}) or {})
        return Document(text=text, title=str(meta.get("title", "")), id=str(meta.get("id", "") or getattr(x, "id", "") or ""),
                        metadata=meta)
    raise TypeError(f"cannot convert {type(x).__name__} to Document")


def as_documents(xs) -> List[Document]:
    return [as_document(x) for x in xs]


# ------------------------------------------------------------------ text utils
def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower case, strip punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_answer(passage: str, answer: str) -> bool:
    """Grounding check: the normalised answer occurs in the normalised passage."""
    a = normalize_answer(answer)
    return bool(a) and a in normalize_answer(passage)
