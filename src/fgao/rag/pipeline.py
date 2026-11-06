"""FGAO-orchestrated RAG (the evaluated v2 RAG policy).

Goals (mandatory, checked mechanically, never taken from the generator's claim)

  answered  (w 0.2)  a non-abstaining answer exists
  grounded  (w 0.3)  the normalised answer occurs in a passage it was generated from
  supported (w 0.5)  the support checker accepted the current answer, reading
                     the passage that contains it

Operations offered to the controller

  answer(src, graded=False)  retrieve from ``src`` (once per query, cached), generate
  answer(src, graded=True)   retrieve, grade relevance, generate from the kept passages
  verify                     support check of the current answer

Each answer operation runs at most once per query (LLM calls are
deterministic, so repeating one cannot help); verify runs once per distinct
answer. Answer operations on sources of the same ``family`` share feedback:
when the KB variants keep failing, FGAO's estimate for all of them drops and it
escalates to the other family (e.g. web search).

Output: the answer only if all three goals are verified, otherwise an
abstention (``answer is None``); ``grounded_answer`` keeps the best grounded but
unverified answer for callers that prefer coverage.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from ..controller import FGAOConfig
from ..workflow import CompiledWorkflow, Step, ToolResult, Workflow, WorkflowResult
from .components import LLMGenerator, LLMGrader, LLMRewriter, LLMSupportChecker, acall
from .documents import Document, contains_answer
from .llms import as_llm
from .retrievers import aretrieve, as_retriever


@dataclass(frozen=True)
class Costs:
    """Declared operation costs (same units as the budget)."""

    llm: float = 2.0
    kb: float = 1.0
    kb_deep: float = 1.5
    web: float = 4.0


@dataclass(frozen=True)
class Priors:
    """Declared effect priors of the evaluated v2 RAG task (fixed a priori, never tuned)."""

    answer_answered: float = 0.8
    answer_grounded: float = 0.7
    graded_answered: float = 0.75
    graded_grounded: float = 0.75
    answer_harms_support: float = 0.9
    answer_produces_unverified: float = 0.8
    verify_supported: float = 0.8


@dataclass(frozen=True)
class Weights:
    answered: float = 0.2
    grounded: float = 0.3
    supported: float = 0.5


@dataclass(frozen=True)
class Source:
    """A retrieval source the controller may choose.

    cost    declared retrieval cost (for ``rewrite=True`` include the rewrite call)
    family  comparability group; sources of one family share feedback
    rewrite rewrite the question into a search query before retrieving
    """

    name: str
    retriever: Any
    k: int = 5
    cost: float = 1.0
    family: str = "kb"
    rewrite: bool = False


def default_sources(kb: Any, web: Any = None, costs: Costs = Costs()) -> List[Source]:
    """The evaluated source layout: KB top-5, KB top-10, rewritten-query KB top-5, web top-5."""
    kb = as_retriever(kb)
    out = [
        Source("kb5", kb, k=5, cost=costs.kb, family="kb"),
        Source("kb10", kb, k=10, cost=costs.kb_deep, family="kb"),
        Source("rw", kb, k=5, cost=costs.llm + costs.kb, family="kb", rewrite=True),
    ]
    if web is not None:
        out.append(Source("web", as_retriever(web), k=5, cost=costs.web, family="web"))
    return out


# =============================================================================
class RAGEpisode:
    """Mutable per-query state the operations act on."""

    def __init__(self, question: str) -> None:
        self.question = question
        self.docs: Dict[str, List[Document]] = {}
        self.graded: Dict[str, List[Document]] = {}
        self.answer: str = ""
        self.answer_docs: List[Document] = []
        self.supported = False
        self.checked: Optional[str] = None  # answer last sent to the support check
        self.tried: Set[Tuple[str, str, int]] = set()
        self.calls: Counter = Counter()
        self.queries: Dict[str, str] = {}

    def grounded(self) -> bool:
        return bool(self.answer) and any(contains_answer(d.text, self.answer) for d in self.answer_docs)

    def support_passage(self) -> Optional[Document]:
        return next((d for d in self.answer_docs if contains_answer(d.text, self.answer)), None)


@dataclass
class RAGResult:
    question: str
    answer: Optional[str]  # None = abstained
    status: str
    values: Dict[str, float]
    grounded_answer: Optional[str]
    evidence: List[Document]
    support_passage: Optional[Document]
    support: Optional[bool]  # support-check result for the final answer; None if it was never checked
    cost: float
    calls: Dict[str, int]
    steps: List[Step] = field(default_factory=list)
    retrieved: Dict[str, List[Document]] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.answer is None

    @property
    def verified(self) -> bool:
        return self.answer is not None

    def trace(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    def explain(self) -> str:
        lines = [f"Q: {self.question}"]
        for s in self.steps:
            d = s.to_dict()
            vals = ", ".join(f"{k}={v:g}" for k, v in d["values"].items())
            lines.append(f"  {d['step']}. {d['action']}  Q={d['Q']:.3f} gain={d['predicted_gain']:.3f} "
                         f"cont={d['continuation']:.3f} cost={d['cost']:g} left={d['budget_left']:g}  -> {vals}"
                         + (f"  [{d['error']}]" if d["error"] else ""))
        lines.append(f"  stop: {self.status}; answer: {self.answer if self.answer is not None else '(abstain)'}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer, "abstained": self.abstained, "status": self.status,
            "values": dict(self.values), "grounded_answer": self.grounded_answer,
            "evidence": [d.to_dict() for d in self.evidence],
            "support_passage": self.support_passage.to_dict() if self.support_passage else None,
            "support": self.support, "cost": self.cost, "calls": dict(self.calls), "trace": self.trace(),
        }


# =============================================================================
class FGAORAG:
    """FGAO v2 as a RAG orchestrator.

    Parameters
    ----------
    sources : list of ``Source`` (see ``default_sources`` / ``from_retrievers``).
    llm     : default LLM for the four operations; anything ``as_llm`` accepts.
    generator, grader, rewriter, support_checker : override single operations.
    costs, priors, weights : declared costs, effect priors and goal weights.
    budget  : per-query budget (default 16).
    config  : ``FGAOConfig`` (default: the evaluated v2 configuration).
    on_error: "raise" (default) propagates an operation's exception (after any
              retries in a ``resilient`` wrapper); "record" charges the
              operation's declared cost, records the error in the trace and
              lets FGAO continue with the remaining operations.
    """

    def __init__(
        self,
        sources: Sequence[Source],
        llm: Any = None,
        *,
        generator: Optional[Callable] = None,
        grader: Optional[Callable] = None,
        rewriter: Optional[Callable] = None,
        support_checker: Optional[Callable] = None,
        costs: Costs = Costs(),
        priors: Priors = Priors(),
        weights: Weights = Weights(),
        budget: float = 16.0,
        config: Optional[FGAOConfig] = None,
        on_error: str = "raise",
    ) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        self.sources = [Source(s.name, as_retriever(s.retriever), s.k, s.cost, s.family, s.rewrite) for s in sources]
        self._src = {s.name: s for s in self.sources}
        need = {"generator": generator, "grader": grader, "support_checker": support_checker}
        if any(s.rewrite for s in self.sources):
            need["rewriter"] = rewriter
        missing = [k for k, v in need.items() if v is None]
        if llm is None and missing:
            raise ValueError(f"llm is required for the default {', '.join(missing)}")
        base = as_llm(llm) if llm is not None else None
        self.llm = base
        self.generator = generator or LLMGenerator(base)
        self.grader = grader or LLMGrader(base)
        self.rewriter = rewriter or (LLMRewriter(base) if base is not None else None)
        self.support_checker = support_checker or LLMSupportChecker(base)
        self.costs, self.priors, self.weights = costs, priors, weights
        self.budget = float(budget)
        self.config = config or FGAOConfig()
        self.on_error = on_error
        self.app: CompiledWorkflow = self._build().compile(self.config, budget=self.budget, on_error=on_error)

    @classmethod
    def from_retrievers(cls, kb: Any, llm: Any = None, *, web: Any = None, costs: Costs = Costs(),
                        **kw: Any) -> "FGAORAG":
        """The evaluated layout over one knowledge base and an optional web/second source."""
        return cls(default_sources(kb, web, costs), llm, costs=costs, **kw)

    # ------------------------------------------------------------ workflow
    def _build(self) -> Workflow:
        P, W, C = self.priors, self.weights, self.costs
        wf = Workflow("fgao-rag")
        wf.add_goal("answered", lambda e: float(bool(e.answer)), weight=W.answered)
        wf.add_goal("grounded", lambda e: float(e.grounded()), weight=W.grounded)
        wf.add_goal("supported", lambda e: float(e.supported and e.grounded()), weight=W.supported)
        for s in self.sources:
            for g in (0, 1):
                wf.add_predicate(f"untried:ans:{s.name}:{g}", (lambda key: lambda e: key not in e.tried)(("ans", s.name, g)))
        wf.add_predicate("unverified", lambda e: bool(e.answer) and e.checked != e.answer)
        for s in self.sources:
            for g in (False, True):
                wf.add_tool(
                    "answer", self._answer, afn=self._aanswer, args={"src": s.name, "graded": g},
                    cost=s.cost + C.llm * (2 if g else 1),
                    gain={"answered": P.graded_answered if g else P.answer_answered,
                          "grounded": P.graded_grounded if g else P.answer_grounded},
                    harm={"supported": P.answer_harms_support},
                    produces={"unverified": P.answer_produces_unverified},
                    requires=[f"untried:ans:{s.name}:{int(g)}"],
                    group=("answer", s.family),
                )
        wf.add_tool("verify", self._verify, afn=self._averify, cost=C.llm, gain={"supported": P.verify_supported},
                    requires=["unverified"], requires_goals=["answered", "grounded"])
        return wf

    def _ensure_docs(self, ep: RAGEpisode, name: str) -> None:
        if name in ep.docs:
            return
        s = self._src[name]
        query = ep.question
        if s.rewrite:
            ep.calls["rewrite"] += 1
            query = self.rewriter(ep.question) or ep.question
            ep.queries[name] = query
        ep.calls[f"retrieve:{name}"] += 1
        ep.docs[name] = list(s.retriever.retrieve(query, s.k))

    def _answer(self, ep: RAGEpisode, src: str, graded: bool) -> Optional[ToolResult]:
        ep.tried.add(("ans", src, int(graded)))
        self._ensure_docs(ep, src)
        docs = ep.docs.get(src, [])
        if graded and docs:
            if src not in ep.graded:
                ep.calls["grade"] += 1
                scores = self.grader(ep.question, docs)
                ep.graded[src] = [d for d, sc in zip(docs, scores) if sc >= 1]
            docs = ep.graded[src]
        if not docs:
            return ToolResult(error="no documents")
        ep.calls["generate"] += 1
        ans = self.generator(ep.question, docs) or ""
        if ans and ans != ep.answer:  # an abstention never overwrites an existing answer
            ep.answer, ep.answer_docs, ep.supported = ans, list(docs), False
        elif ans and ans == ep.answer and not ep.grounded():
            ep.answer_docs = list(docs)
        return None

    def _verify(self, ep: RAGEpisode) -> Optional[ToolResult]:
        hit = ep.support_passage()
        if hit is None or not ep.answer:
            return ToolResult(error="nothing to verify")
        ep.calls["verify"] += 1
        ep.checked = ep.answer
        ep.supported = bool(self.support_checker(ep.question, hit, ep.answer))
        return None

    # async twins (identical logic; components are awaited natively when they support it)
    async def _aensure_docs(self, ep: RAGEpisode, name: str) -> None:
        if name in ep.docs:
            return
        s = self._src[name]
        query = ep.question
        if s.rewrite:
            ep.calls["rewrite"] += 1
            query = (await acall(self.rewriter, ep.question)) or ep.question
            ep.queries[name] = query
        ep.calls[f"retrieve:{name}"] += 1
        ep.docs[name] = list(await aretrieve(s.retriever, query, s.k))

    async def _aanswer(self, ep: RAGEpisode, src: str, graded: bool) -> Optional[ToolResult]:
        ep.tried.add(("ans", src, int(graded)))
        await self._aensure_docs(ep, src)
        docs = ep.docs.get(src, [])
        if graded and docs:
            if src not in ep.graded:
                ep.calls["grade"] += 1
                scores = await acall(self.grader, ep.question, docs)
                ep.graded[src] = [d for d, sc in zip(docs, scores) if sc >= 1]
            docs = ep.graded[src]
        if not docs:
            return ToolResult(error="no documents")
        ep.calls["generate"] += 1
        ans = (await acall(self.generator, ep.question, docs)) or ""
        if ans and ans != ep.answer:
            ep.answer, ep.answer_docs, ep.supported = ans, list(docs), False
        elif ans and ans == ep.answer and not ep.grounded():
            ep.answer_docs = list(docs)
        return None

    async def _averify(self, ep: RAGEpisode) -> Optional[ToolResult]:
        hit = ep.support_passage()
        if hit is None or not ep.answer:
            return ToolResult(error="nothing to verify")
        ep.calls["verify"] += 1
        ep.checked = ep.answer
        ep.supported = bool(await acall(self.support_checker, ep.question, hit, ep.answer))
        return None

    # ------------------------------------------------------------- running
    @staticmethod
    def _question(x: Union[str, Dict[str, Any]]) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("question", "query", "input"):
                if k in x:
                    return str(x[k])
        raise TypeError("input must be a question string or a dict with 'question'")

    def _result(self, ep: RAGEpisode, wr: WorkflowResult) -> RAGResult:
        grounded = ep.grounded()
        return RAGResult(
            question=ep.question,
            answer=ep.answer if wr.success else None,
            status=wr.status,
            values=dict(wr.values),
            grounded_answer=ep.answer if grounded else None,
            evidence=list(ep.answer_docs),
            support_passage=ep.support_passage() if grounded else None,
            support=(ep.supported if ep.answer and ep.checked == ep.answer else None),
            cost=wr.cost,
            calls=dict(ep.calls),
            steps=list(wr.steps),
            retrieved={k: list(v) for k, v in ep.docs.items()},
        )

    def stream(self, question: Union[str, Dict[str, Any]], *, seed: int = 0, budget: Optional[float] = None,
               callbacks: Sequence[Callable[[Step], None]] = ()) -> Iterator[Union[Step, RAGResult]]:
        """Yield every executed ``Step`` as it happens, then the ``RAGResult``."""
        ep = RAGEpisode(self._question(question))
        for item in self.app.stream(ep, seed=seed, budget=budget, callbacks=callbacks):
            yield self._result(ep, item) if isinstance(item, WorkflowResult) else item

    def invoke(self, question: Union[str, Dict[str, Any]], *, seed: int = 0, budget: Optional[float] = None,
               callbacks: Sequence[Callable[[Step], None]] = ()) -> RAGResult:
        out = None
        for out in self.stream(question, seed=seed, budget=budget, callbacks=callbacks):
            pass
        return out  # type: ignore[return-value]

    __call__ = invoke

    def batch(self, questions: Sequence[Union[str, Dict[str, Any]]], *, max_concurrency: int = 8,
              **kw: Any) -> List[RAGResult]:
        """Answer many questions concurrently (threads); results keep input order."""
        if max_concurrency <= 1:
            return [self.invoke(q, **kw) for q in questions]
        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            return list(ex.map(lambda q: self.invoke(q, **kw), questions))

    async def astream(self, question: Union[str, Dict[str, Any]], *, seed: int = 0, budget: Optional[float] = None,
                      callbacks: Sequence[Callable[[Step], None]] = ()) -> AsyncIterator[Union[Step, RAGResult]]:
        """Native async ``stream``: same decisions; LLM and retriever calls are awaited."""
        ep = RAGEpisode(self._question(question))
        async for item in self.app.astream(ep, seed=seed, budget=budget, callbacks=callbacks):
            yield self._result(ep, item) if isinstance(item, WorkflowResult) else item

    async def ainvoke(self, question: Union[str, Dict[str, Any]], *, seed: int = 0, budget: Optional[float] = None,
                      callbacks: Sequence[Callable[[Step], None]] = ()) -> RAGResult:
        out = None
        async for out in self.astream(question, seed=seed, budget=budget, callbacks=callbacks):
            pass
        return out  # type: ignore[return-value]

    async def abatch(self, questions: Sequence[Union[str, Dict[str, Any]]], *, max_concurrency: int = 64,
                     **kw: Any) -> List[RAGResult]:
        """Answer many questions concurrently on the event loop; results keep input order."""
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def one(q):
            async with sem:
                return await self.ainvoke(q, **kw)

        return list(await asyncio.gather(*(one(q) for q in questions)))

    # -------------------------------------------------------- integrations
    def as_runnable(self):
        """A LangChain ``Runnable``: input a question (str or {'question': ...}), output ``RAGResult.to_dict()``."""
        from ..integrations.langchain import rag_runnable

        return rag_runnable(self)

    def as_tool(self, name: str = "fgao_rag", description: Optional[str] = None):
        """A LangChain tool an agent can call with a question."""
        from ..integrations.langchain import rag_tool

        return rag_tool(self, name, description)

    def as_node(self, *, question_key: str = "question", answer_key: str = "answer",
                result_key: Optional[str] = "fgao"):
        """A LangGraph node: reads ``state[question_key]``, writes the answer (and the full result)."""
        from ..integrations.langgraph import rag_node

        return rag_node(self, question_key=question_key, answer_key=answer_key, result_key=result_key)
