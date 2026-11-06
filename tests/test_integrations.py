from types import SimpleNamespace

import pytest

from fgao import FGAORAG, Workflow
from fgao.integrations.huggingface import HFInferenceLLM, hf_embedder
from fgao.integrations.langgraph import workflow_node
from fgao.integrations.providers import AnthropicLLM, OpenAILLM
from fgao.rag import DenseRetriever

from test_rag import KB, WEB, Scripted


def test_hf_inference_llm_with_client():
    seen = {}

    class Client:
        def chat_completion(self, messages, model, max_tokens, **kw):
            seen.update(messages=messages, model=model, kw=kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" Paris \n"))])

        def feature_extraction(self, text, model):
            return [[1.0, float("France" in text)], [3.0, float("France" in text)]]  # token embeddings

    llm = HFInferenceLLM("org/model", client=Client(), temperature=0)
    assert llm.complete("hi") == "Paris" and seen["model"] == "org/model" and seen["kw"] == {"temperature": 0}
    assert seen["messages"][0]["role"] == "system"
    emb = hf_embedder("org/emb", client=Client())
    assert emb(["France"]) == [[2.0, 1.0]]
    assert DenseRetriever(KB, emb).retrieve("France", 1)[0].text == KB[0]


def test_provider_llms_with_clients():
    oc = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]))))
    assert OpenAILLM("m", client=oc).complete("x") == "ok"
    ac = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(text="ok")])))
    assert AnthropicLLM("m", client=ac).complete("x") == "ok"


def test_workflow_node_plain():
    wf = Workflow("inc").add_goal("done", lambda s: s.get("n", 0) >= 2)
    wf.add_tool("inc", lambda s: {"n": s.get("n", 0) + 1}, gain={"done": 0.5})
    node = workflow_node(wf.compile(budget=10))
    out = node({"n": 0})
    assert out["n"] == 2 and out["fgao"]["success"]


def test_langchain_llm_retriever_runnable_tool():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document as LCDocument
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnableLambda
    from langchain_core.vectorstores import InMemoryVectorStore
    from fgao.integrations.langchain import LangChainLLM, LangChainRetriever

    assert LangChainLLM(FakeListChatModel(responses=["yes"])).complete("x") == "yes"

    class R(BaseRetriever):
        docs: list

        def _get_relevant_documents(self, query, *, run_manager=None):
            return [d for d in self.docs if any(w in d.page_content for w in query.split())]

    kb = R(docs=[LCDocument(page_content=t, metadata={"title": "Atlas"}) for t in KB])
    vs = InMemoryVectorStore.from_texts(WEB, DeterministicFakeEmbedding(size=16))
    assert len(LangChainRetriever(vs).retrieve("Canada", 2)) == 2
    # LangChain objects are accepted directly (auto-adapted)
    app = FGAORAG.from_retrievers(kb, RunnableLambda(Scripted()), web=WEB)
    r = app.invoke("What is the capital of France?")
    assert r.answer == "Paris" and r.support_passage.title == "Atlas"

    chain = app.as_runnable() | RunnableLambda(lambda d: d["answer"])
    assert chain.invoke({"question": "What is the capital of Italy?"}) == "Rome"
    assert chain.batch(["What is the capital of Italy?", "What is the capital of Spain?"]) == ["Rome", None]
    tool = app.as_tool()
    out = tool.invoke({"question": "What is the capital of Canada?"})
    assert out["answer"] == "Ottawa" and "Ottawa" in out["evidence"]


def test_langgraph_node():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langgraph")
    from typing import Optional, TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        question: str
        answer: Optional[str]
        fgao: dict

    app = FGAORAG.from_retrievers(KB, Scripted(), web=WEB)
    g = StateGraph(S)
    g.add_node("rag", app.as_node())
    g.add_edge(START, "rag")
    g.add_edge("rag", END)
    out = g.compile().invoke({"question": "What is the capital of Germany?"})
    assert out["answer"] == "Berlin" and out["fgao"]["status"] == "SUCCESS"
