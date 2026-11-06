import asyncio
import json
import re

import pytest

from fgao import (BM25Retriever, CachedLLM, DenseRetriever, Document, FGAOConfig, FGAORAG, FunctionRetriever,
                  RAGResult, Source, Status, Step)
from fgao.rag import RAGEpisode, as_document, contains_answer, normalize_answer
from fgao.rag import prompts as P

KB = ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Rome is the capital of Italy."]
WEB = ["Canberra is the capital of Australia.", "Ottawa is the capital of Canada."]


class Scripted:
    """Answers with a capital that occurs in the passages; scripted support verdicts."""

    def __init__(self, support="yes", fabricate=False):
        self.support, self.fabricate, self.prompts = support, fabricate, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Answer the question"):
            body = prompt.split("Passages:")[1].split("Question:")[0]
            q = prompt.rsplit("Question: ", 1)[1]
            country = re.search(r"capital of (\w+)", q).group(1)
            m = re.search(r"(\w+) is the capital of " + country, body)
            if self.fabricate:
                return "Atlantis"
            return m.group(1) if m else "unanswerable"
        if prompt.startswith("Rate how useful"):
            return "\n".join(f"[{i + 1}] 2" for i in range(prompt.count("\n[")))
        if prompt.startswith("Rewrite"):
            return "capital city query"
        if prompt.startswith("Does the passage"):
            return self.support
        raise AssertionError(prompt)


def rag(llm=None, **kw):
    return FGAORAG.from_retrievers(KB, llm or Scripted(), web=WEB, **kw)


def test_answers_from_kb_and_verifies():
    r = rag().invoke("What is the capital of France?")
    assert r.answer == "Paris" and r.status == Status.SUCCESS and r.verified and not r.abstained
    assert [str(s.action) for s in r.steps] == ["answer(src=kb5, graded=False)", "verify()"]
    assert r.support is True and r.support_passage.text == KB[0] and r.cost == 5.0
    assert r.calls == {"retrieve:kb5": 1, "generate": 1, "verify": 1}
    assert r.values == {"answered": 1.0, "grounded": 1.0, "supported": 1.0}


def test_abstains_when_support_check_rejects():
    r = rag(Scripted(support="no")).invoke("What is the capital of France?")
    assert r.answer is None and r.abstained and r.grounded_answer == "Paris" and r.support is False
    assert r.status in (Status.STALLED, Status.BUDGET_EXHAUSTED)


def test_escalates_to_web_when_kb_lacks_evidence():
    r = rag().invoke("What is the capital of Australia?")
    assert r.answer == "Canberra"
    acts = [str(s.action) for s in r.steps]
    assert any("src=web" in a for a in acts) and acts[-1] == "verify()"


def test_ungrounded_answer_is_never_returned():
    r = rag(Scripted(fabricate=True)).invoke("What is the capital of France?")
    assert r.answer is None and r.grounded_answer is None and r.support is None
    assert "verify" not in r.calls


def test_budget_limits_work():
    r = rag().invoke("What is the capital of France?", budget=2.0)
    assert r.answer is None and r.cost <= 2.0


def test_input_forms_and_serialisation():
    app = rag()
    for x in ("What is the capital of Italy?", {"question": "What is the capital of Italy?"}, {"query": "What is the capital of Italy?"}):
        assert app.invoke(x).answer == "Rome"
    with pytest.raises(TypeError):
        app.invoke(42)
    d = app("What is the capital of Italy?").to_dict()
    json.dumps(d)
    assert d["answer"] == "Rome" and d["trace"][0]["action"].startswith("answer(")
    assert "verify()" in app.invoke("What is the capital of Italy?").explain()


def test_stream_batch_async():
    app = rag()
    items = list(app.stream("What is the capital of Germany?"))
    assert all(isinstance(i, Step) for i in items[:-1]) and isinstance(items[-1], RAGResult)
    qs = ["What is the capital of France?", "What is the capital of Canada?", "What is the capital of Spain?"]
    rs = app.batch(qs, max_concurrency=3)
    assert [r.answer for r in rs] == ["Paris", "Ottawa", None]
    assert [r.answer for r in asyncio.run(app.abatch(qs))] == ["Paris", "Ottawa", None]
    assert asyncio.run(app.ainvoke(qs[0])).answer == "Paris"

    async def collect():
        return [x async for x in app.astream(qs[0])]

    assert isinstance(asyncio.run(collect())[-1], RAGResult)


def test_custom_components_without_llm():
    kb = BM25Retriever(KB)
    app = FGAORAG([Source("kb", kb, k=3, cost=1.0)],
                  generator=lambda q, docs: "Rome" if any("Rome" in d.text for d in docs) else "",
                  grader=lambda q, docs: [2] * len(docs),
                  support_checker=lambda q, passage, a: a in passage.text)
    r = app.invoke("capital of Italy")
    assert r.answer == "Rome" and len(app.app.actions) == 3
    with pytest.raises(ValueError, match="llm is required"):
        FGAORAG.from_retrievers(KB)  # the rewrite source needs a rewriter


def test_source_layout_and_validation():
    app = rag()
    assert [str(a) for a in app.app.actions][-1] == "verify()" and len(app.app.actions) == 9
    with pytest.raises(ValueError):
        FGAORAG([], Scripted())
    with pytest.raises(ValueError):
        FGAORAG([Source("a", KB), Source("a", KB)], Scripted())
    assert len(FGAORAG.from_retrievers(KB, Scripted()).app.actions) == 7  # no web source


def test_rewrite_source_uses_rewritten_query():
    llm = Scripted()
    ep = RAGEpisode("What is the capital of France?")
    app = rag(llm)
    app._ensure_docs(ep, "rw")
    assert ep.queries["rw"] == "capital city query" and ep.calls["rewrite"] == 1


def test_cached_llm(tmp_path):
    inner = Scripted()
    path = str(tmp_path / "cache.json")
    c = CachedLLM(inner, path)
    app = rag(c)
    app.invoke("What is the capital of France?")
    n = len(inner.prompts)
    app.invoke("What is the capital of France?")
    assert len(inner.prompts) == n and c.hits == n
    assert len(CachedLLM(inner, path).cache) == n


def test_retrievers():
    bm = BM25Retriever(KB + WEB)
    assert bm.retrieve("capital of Canada", 1)[0].text == WEB[1]
    fr = FunctionRetriever(lambda q, k: [{"text": "x", "title": "T", "id": "1", "url": "u"}] * 5)
    d = fr.retrieve("q", 2)
    assert len(d) == 2 and d[0].metadata == {"url": "u"} and d[0].title == "T"
    emb = lambda texts: [[t.count("France"), t.count("Germany"), 1.0] for t in texts]
    dr = DenseRetriever(KB, emb)
    assert dr.retrieve("Germany", 1)[0].text == KB[1]


def test_text_utils_and_prompts():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert contains_answer("It is the Eiffel tower.", "eiffel Tower") and not contains_answer("x", "")
    assert as_document("t").text == "t" and as_document({"page_content": "p"}).text == "p"
    assert P.parse_answer("Answer: Paris.") == "Paris" and P.parse_answer("Unanswerable") == ""
    assert P.parse_grades("[1] 2\n[2] 0\n[3] 1", 3) == [2, 0, 1]
    assert P.parse_yes(" Yes.") and not P.parse_yes("no")
    assert "[1] (T) body" in P.generate_prompt("q", [Document("body", "T")])
    assert "[1] body" in P.generate_prompt("q", [Document("body")])


def test_config_passthrough():
    app = rag(config=FGAOConfig(lookahead="eq16"))
    assert app.invoke("What is the capital of France?").answer == "Paris"
