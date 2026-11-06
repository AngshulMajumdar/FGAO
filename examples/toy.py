"""A tiny corpus and a rule-based stand-in LLM so the examples run offline.

ToyLLM understands the four FGAO RAG prompts and the fact pattern
"<answer> is the <relation> of <entity>." It exists only to show the
plumbing; use a real model (see quickstart.py) for real questions.
"""
import re

KB = [
    {"title": "France", "text": "Paris is the capital of France. The Seine flows through the city."},
    {"title": "Germany", "text": "Berlin is the capital of Germany. It has around 3.7 million inhabitants."},
    {"title": "Italy", "text": "Rome is the capital of Italy. The Tiber runs through it."},
    {"title": "Japan", "text": "Japan is an island country in East Asia. Mount Fuji is its highest peak."},
]
WEB = [
    {"title": "Japan (web)", "text": "Tokyo is the capital of Japan."},
    {"title": "Kenya (web)", "text": "Nairobi is the capital of Kenya."},
    {"title": "Peru (web)", "text": "Lima is the capital of Peru."},
]

_FACT = re.compile(r"([A-Z][\w.]*(?: [A-Z][\w.]*)*) is the (\w+) of (\w+)")


class ToyLLM:
    def complete(self, prompt: str) -> str:
        if prompt.startswith("Answer the question"):
            body, q = prompt.split("Passages:", 1)[1].rsplit("Question:", 1)
            m = re.search(r"(?:the )?(\w+) of (\w+)", q)
            for ans, rel, ent in _FACT.findall(body):
                if m and rel == m.group(1) and ent == m.group(2):
                    return ans
            return "unanswerable"
        if prompt.startswith("Rate how useful"):
            q = prompt.split("Question:", 1)[1].split("Passages:", 1)[0].lower()
            lines = [ln for ln in prompt.split("Passages:", 1)[1].splitlines() if ln.startswith("[")]
            words = set(re.findall(r"[a-z]{4,}", q))
            return "\n".join(f"[{i + 1}] {2 if words & set(re.findall(r'[a-z]{4,}', ln.lower())) else 0}"
                             for i, ln in enumerate(lines))
        if prompt.startswith("Rewrite the question"):
            q = prompt.rsplit("Question:", 1)[1].split("\n", 1)[0]
            return " ".join(w for w in re.findall(r"[A-Za-z]+", q) if len(w) > 3)
        if prompt.startswith("Does the passage"):
            passage = prompt.split("Passage:", 1)[1].split("Question:", 1)[0]
            answer = prompt.split("Answer:", 1)[1].split("\n", 1)[0].strip()
            q = prompt.split("Question:", 1)[1].split("\n", 1)[0]
            m = re.search(r"(\w+) of (\w+)", q)
            ok = m and f"{answer} is the {m.group(1)} of {m.group(2)}" in passage
            return "yes" if ok else "no"
        return ""
