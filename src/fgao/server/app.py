"""HTTP service around an ``FGAORAG`` instance (FastAPI).

Endpoints

  GET  /health                 liveness
  GET  /ready                  readiness
  GET  /v1/info                version, sources, costs, budget, controller configuration
  POST /v1/answer              {"question", "budget"?, "seed"?, "trace"?, "documents"?, "web_documents"?}
                               -> RAGResult as JSON; with "documents", FGAO answers from those passages
  POST /v1/answer/stream       same body; Server-Sent Events: "step" per decision, then "result"
  POST /v1/batch               {"questions": [...], ...}  -> {"results": [...]}
  GET  /metrics                Prometheus metrics
  GET  /docs, /openapi.json    interactive API docs

Operational behaviour

* Admission control: at most ``max_inflight`` queries run at once; up to
  ``max_queue`` more wait; beyond that the server answers 429 with Retry-After.
* Each query has a deadline (``request_timeout`` s): 504 on expiry, and the
  in-flight model call is cancelled.
* A failing dependency (after retries) answers 502 with the error type,
  unless the pipeline was built with ``on_error="record"``.
* Optional API keys: ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
* Every response carries ``X-Request-ID`` (echoed if the client sent one), and
  one JSON log line per query is written to the ``fgao.server`` logger.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union

from .. import __version__
from ..rag.pipeline import FGAORAG, RAGResult
from ..resilience import _Resilient
from .metrics import Registry

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

log = logging.getLogger("fgao.server")


DocIn = Union[str, Dict[str, Any]]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    budget: Optional[float] = Field(None, gt=0)
    seed: int = 0
    trace: bool = True
    documents: Optional[List[DocIn]] = Field(None, description="answer from these passages instead of the configured KB")
    web_documents: Optional[List[DocIn]] = Field(None, description="optional second source for the request documents")


class BatchRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    budget: Optional[float] = Field(None, gt=0)
    seed: int = 0
    trace: bool = False
    documents: Optional[List[DocIn]] = None
    web_documents: Optional[List[DocIn]] = None


class Overloaded(Exception):
    pass


class Admission:
    """Bounded concurrency with a bounded wait queue (per event loop / worker process)."""

    def __init__(self, max_inflight: int, max_queue: int) -> None:
        self.max_inflight, self.max_queue = max_inflight, max_queue
        self.inflight = 0
        self.waiting = 0
        self._sem: Optional[asyncio.Semaphore] = None

    def _s(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_inflight)
        return self._sem

    def check(self, n: int = 1) -> None:
        free = self.max_inflight - self.inflight
        if n > free and self.waiting + n - max(free, 0) > self.max_queue:
            raise Overloaded()

    async def acquire(self) -> None:
        self.check()
        self.waiting += 1
        try:
            await self._s().acquire()
        finally:
            self.waiting -= 1
        self.inflight += 1

    def release(self) -> None:
        self.inflight -= 1
        self._s().release()


def _resilient_parts(rag: FGAORAG) -> List[_Resilient]:
    seen, out = set(), []
    cands = [rag.llm] + [s.retriever for s in rag.sources]
    for comp in (rag.generator, rag.grader, rag.rewriter, rag.support_checker):
        cands.append(getattr(comp, "llm", None))
    for c in cands:
        if isinstance(c, _Resilient) and id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def create_app(
    rag: FGAORAG,
    *,
    api_keys: Optional[Iterable[str]] = None,
    max_inflight: int = 256,
    max_queue: int = 1024,
    request_timeout: float = 60.0,
    max_batch: int = 256,
    max_budget: Optional[float] = None,
    max_documents: int = 500,
    title: str = "FGAO RAG",
):
    """Build the FastAPI application. ``max_budget`` caps per-request budgets (default: the pipeline budget x 4)."""
    keys = {k for k in (api_keys or []) if k}
    cap = float(max_budget if max_budget is not None else rag.budget * 4)
    adm = Admission(max_inflight, max_queue)
    reg = Registry()
    m_http = reg.counter("fgao_http_requests_total", "HTTP requests by route and status code")
    m_lat = reg.histogram("fgao_request_duration_seconds", "Query latency in seconds",
                          (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
    m_out = reg.counter("fgao_queries_total", "Queries by outcome (verified, abstained, timeout, error)")
    m_stop = reg.counter("fgao_stop_status_total", "Queries by FGAO stop status")
    m_cost = reg.histogram("fgao_query_cost", "Budget spent per query", (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 64))
    m_ops = reg.counter("fgao_operation_calls_total", "Operation calls (generate, grade, rewrite, verify, retrieve:*)")
    m_steps = reg.counter("fgao_steps_total", "Controller steps executed")
    m_inflight = reg.gauge("fgao_inflight", "Queries running")
    m_queue = reg.gauge("fgao_queue", "Queries waiting for a slot")
    parts = _resilient_parts(rag)

    def collect():
        m_inflight.set(adm.inflight)
        m_queue.set(adm.waiting)
        for p in parts:
            for k, v in p.stats.as_dict().items():
                yield (f"fgao_dependency_{k}_total", "counter", f"Dependency {k} (resilient wrappers)", {"name": p.name}, v)

    reg.collectors.append(lambda: list(collect()))

    app = FastAPI(title=title, version=__version__,
                  description="Training-free FGAO orchestration of retrieval-augmented generation.")

    def auth(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> None:
        if not keys:
            return
        token = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if token not in keys:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.rid = rid
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        m_http.inc(route=getattr(route, "path", request.url.path), code=str(resp.status_code))
        return resp

    def _budget(b: Optional[float]) -> Optional[float]:
        if b is not None and b > cap:
            raise HTTPException(status_code=422, detail=f"budget exceeds the server limit {cap:g}")
        return b

    def _payload(r: RAGResult, trace: bool) -> Dict[str, Any]:
        d = r.to_dict()
        if not trace:
            d.pop("trace", None)
        return d

    def _record(r: RAGResult, seconds: float, rid: str) -> None:
        m_out.inc(outcome="verified" if r.answer is not None else "abstained")
        m_stop.inc(status=r.status)
        m_lat.observe(seconds)
        m_cost.observe(r.cost)
        m_steps.inc(len(r.steps))
        for k, v in r.calls.items():
            m_ops.inc(v, op=k)
        log.info(json.dumps({"event": "query", "request_id": rid, "status": r.status, "verified": r.answer is not None,
                             "cost": r.cost, "steps": len(r.steps), "seconds": round(seconds, 4)}))

    def _pipeline(documents, web_documents) -> FGAORAG:
        """The configured pipeline, or one over the request's own passages (same LLM, costs, priors, controller)."""
        if not documents:
            return rag
        if len(documents) > max_documents or (web_documents and len(web_documents) > max_documents):
            raise HTTPException(status_code=422, detail=f"at most {max_documents} documents per source")
        if rag.llm is None:
            raise HTTPException(status_code=422, detail="request documents need a pipeline with a default LLM")
        return FGAORAG.from_retrievers(kb=list(documents), web=list(web_documents) if web_documents else None,
                                       llm=rag.llm, costs=rag.costs, priors=rag.priors, weights=rag.weights,
                                       budget=rag.budget, config=rag.config, on_error=rag.on_error)

    async def _run(q: str, budget: Optional[float], seed: int, rid: str, pipe: Optional[FGAORAG] = None) -> RAGResult:
        t0 = time.perf_counter()
        pipe = pipe or rag
        try:
            r = await asyncio.wait_for(pipe.ainvoke(q, budget=budget, seed=seed), request_timeout)
        except asyncio.TimeoutError:
            m_out.inc(outcome="timeout")
            m_lat.observe(time.perf_counter() - t0)
            log.warning(json.dumps({"event": "timeout", "request_id": rid}))
            raise
        except Exception as e:  # noqa: BLE001
            m_out.inc(outcome="error")
            m_lat.observe(time.perf_counter() - t0)
            log.error(json.dumps({"event": "error", "request_id": rid, "error": type(e).__name__, "detail": str(e)[:500]}))
            raise
        _record(r, time.perf_counter() - t0, rid)
        return r

    def _overloaded() -> JSONResponse:
        return JSONResponse({"detail": "server overloaded, retry later"}, status_code=429, headers={"Retry-After": "1"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready", "inflight": adm.inflight, "queued": adm.waiting}

    @app.get("/v1/info", dependencies=[Depends(auth)])
    async def info():
        return {
            "version": __version__,
            "budget": rag.budget, "max_budget": cap,
            "sources": [{"name": s.name, "k": s.k, "cost": s.cost, "family": s.family, "rewrite": s.rewrite}
                        for s in rag.sources],
            "costs": dataclasses.asdict(rag.costs),
            "weights": dataclasses.asdict(rag.weights),
            "priors": dataclasses.asdict(rag.priors),
            "controller": dataclasses.asdict(rag.config),
            "operations": [str(a) for a in rag.app.actions],
            "on_error": rag.on_error,
            "limits": {"max_inflight": max_inflight, "max_queue": max_queue, "request_timeout": request_timeout,
                       "max_batch": max_batch},
        }

    @app.post("/v1/answer", dependencies=[Depends(auth)])
    async def answer(req: AnswerRequest, request: Request):
        budget = _budget(req.budget)
        pipe = _pipeline(req.documents, req.web_documents)
        try:
            await adm.acquire()
        except Overloaded:
            return _overloaded()
        try:
            r = await _run(req.question, budget, req.seed, request.state.rid, pipe)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"query exceeded {request_timeout:g} s")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}"[:500])
        finally:
            adm.release()
        return _payload(r, req.trace)

    @app.post("/v1/batch", dependencies=[Depends(auth)])
    async def batch(req: BatchRequest, request: Request):
        if len(req.questions) > max_batch:
            raise HTTPException(status_code=422, detail=f"at most {max_batch} questions per batch")
        budget = _budget(req.budget)
        pipe = _pipeline(req.documents, req.web_documents)
        try:
            adm.check(len(req.questions))
        except Overloaded:
            return _overloaded()

        async def one(i: int, q: str) -> Dict[str, Any]:
            await adm.acquire()  # admitted above; this only waits for a slot
            try:
                return _payload(await _run(q, budget, req.seed, f"{request.state.rid}:{i}", pipe), req.trace)
            except asyncio.TimeoutError:
                return {"question": q, "error": "timeout"}
            except Exception as e:  # noqa: BLE001
                return {"question": q, "error": f"{type(e).__name__}: {e}"[:500]}
            finally:
                adm.release()

        results = await asyncio.gather(*(one(i, q) for i, q in enumerate(req.questions)))
        return {"results": list(results)}

    @app.post("/v1/answer/stream", dependencies=[Depends(auth)])
    async def stream(req: AnswerRequest, request: Request):
        budget = _budget(req.budget)
        pipe = _pipeline(req.documents, req.web_documents)
        try:
            await adm.acquire()
        except Overloaded:
            return _overloaded()
        rid = request.state.rid
        released = [False]

        def release_once() -> None:
            if not released[0]:
                released[0] = True
                adm.release()

        async def events():
            t0 = time.perf_counter()
            agen = pipe.astream(req.question, budget=budget, seed=req.seed).__aiter__()
            try:
                while True:
                    left = request_timeout - (time.perf_counter() - t0)
                    try:
                        item = await asyncio.wait_for(agen.__anext__(), max(left, 0.001))
                    except StopAsyncIteration:
                        break
                    if isinstance(item, RAGResult):
                        _record(item, time.perf_counter() - t0, rid)
                        yield f"event: result\ndata: {json.dumps(_payload(item, req.trace))}\n\n"
                    else:
                        yield f"event: step\ndata: {json.dumps(item.to_dict())}\n\n"
            except asyncio.TimeoutError:
                m_out.inc(outcome="timeout")
                yield f"event: error\ndata: {json.dumps({'error': 'timeout'})}\n\n"
            except Exception as e:  # noqa: BLE001
                m_out.inc(outcome="error")
                yield f"event: error\ndata: {json.dumps({'error': f'{type(e).__name__}: {e}'[:500]})}\n\n"
            finally:
                release_once()

        from starlette.background import BackgroundTask

        return StreamingResponse(events(), media_type="text/event-stream", background=BackgroundTask(release_once),
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(reg.render(), media_type="text/plain; version=0.0.4")

    app.state.rag = rag
    app.state.admission = adm
    app.state.registry = reg
    return app
