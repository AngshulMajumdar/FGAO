import asyncio
import json
import os
import sys

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fgao import FGAORAG  # noqa: E402
from fgao.resilience import resilient  # noqa: E402
from fgao.server import build_from_config, create_app  # noqa: E402
from fgao.server.cli import main as cli_main  # noqa: E402

from test_rag import KB, WEB, Scripted  # noqa: E402


def client(**kw):
    rag = kw.pop("rag", None) or FGAORAG.from_retrievers(KB, resilient(Scripted(), attempts=2), web=WEB)
    return TestClient(create_app(rag, **kw))


def test_answer_info_health_metrics():
    c = client()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/v1/answer", json={"question": "What is the capital of France?"}, headers={"X-Request-ID": "abc"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "abc"
    d = r.json()
    assert d["answer"] == "Paris" and d["status"] == "SUCCESS" and d["trace"][-1]["action"] == "verify()"
    assert "trace" not in c.post("/v1/answer", json={"question": "What is the capital of Spain?", "trace": False}).json()
    info = c.get("/v1/info").json()
    assert [s["name"] for s in info["sources"]] == ["kb5", "kb10", "rw", "web"] and info["controller"]["nu"] == 4.0
    m = c.get("/metrics").text
    assert 'fgao_queries_total{outcome="verified"} 1' in m and 'fgao_queries_total{outcome="abstained"} 1' in m
    assert 'fgao_dependency_calls_total{name="llm"}' in m and "fgao_request_duration_seconds_bucket" in m
    assert 'fgao_http_requests_total{code="200",route="/v1/answer"} 2' in m


def test_validation_and_budget_cap():
    c = client()
    assert c.post("/v1/answer", json={"question": ""}).status_code == 422
    assert c.post("/v1/answer", json={"question": "q", "budget": 1000}).status_code == 422
    r = c.post("/v1/answer", json={"question": "What is the capital of France?", "budget": 2})
    assert r.status_code == 200 and r.json()["answer"] is None


def test_batch_and_stream():
    c = client(max_batch=3)
    r = c.post("/v1/batch", json={"questions": ["What is the capital of France?", "What is the capital of Canada?"]})
    assert [x["answer"] for x in r.json()["results"]] == ["Paris", "Ottawa"]
    assert c.post("/v1/batch", json={"questions": ["q"] * 4}).status_code == 422
    with c.stream("POST", "/v1/answer/stream", json={"question": "What is the capital of Canada?"}) as s:
        body = "".join(s.iter_text())
    events = [blk.split("\n")[0][7:] for blk in body.strip().split("\n\n")]
    assert events[-1] == "result" and set(events[:-1]) == {"step"}
    final = json.loads(body.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert final["answer"] == "Ottawa"


def test_auth():
    c = client(api_keys=["k1"])
    assert c.post("/v1/answer", json={"question": "q"}).status_code == 401
    assert c.post("/v1/answer", json={"question": "What is the capital of France?"},
                  headers={"Authorization": "Bearer k1"}).status_code == 200
    assert c.get("/v1/info", headers={"X-API-Key": "k1"}).status_code == 200
    assert c.get("/health").status_code == 200  # probes stay open


def test_timeout_error_and_overload():
    class Slow:
        def complete(self, p):  # pragma: no cover
            return ""

        async def acomplete(self, p):
            await asyncio.sleep(2)
            return "unanswerable"

    c = client(rag=FGAORAG.from_retrievers(KB, Slow(), web=WEB), request_timeout=0.2)
    assert c.post("/v1/answer", json={"question": "What is the capital of France?"}).status_code == 504
    assert 'outcome="timeout"' in c.get("/metrics").text

    class Down:
        def complete(self, p):
            raise ConnectionError("provider down")

    c = client(rag=FGAORAG.from_retrievers(KB, Down(), web=WEB))
    r = c.post("/v1/answer", json={"question": "What is the capital of France?"})
    assert r.status_code == 502 and "ConnectionError" in r.json()["detail"]
    assert 'outcome="error"' in c.get("/metrics").text

    app = create_app(FGAORAG.from_retrievers(KB, Scripted(), web=WEB), max_inflight=1, max_queue=0)
    app.state.admission.inflight = 1  # simulate a busy server
    r = TestClient(app).post("/v1/answer", json={"question": "q"})
    assert r.status_code == 429 and r.headers["Retry-After"] == "1"


def test_config_and_cli(tmp_path, capsys, monkeypatch):
    kb = tmp_path / "kb.jsonl"
    kb.write_text("\n".join(json.dumps({"text": t, "title": "T"}) for t in KB))
    web = tmp_path / "web.txt"
    web.write_text("\n\n".join(WEB))
    mod = tmp_path / "mymod.py"
    mod.write_text("import sys\nsys.path.insert(0, %r)\nfrom test_rag import Scripted\n"
                   "def make_llm():\n    return Scripted()\n" % os.path.dirname(__file__))
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = {"llm": {"provider": "factory", "factory": "mymod:make_llm"}, "kb": {"type": "bm25", "path": "kb.jsonl"},
           "web": "web.txt", "resilience": {"attempts": 2, "backoff": 0}, "budget": 16, "on_error": "record",
           "server": {"max_inflight": 8, "api_keys_env": "MY_KEYS"}}
    path = tmp_path / "fgao.json"
    path.write_text(json.dumps(cfg))
    rag, settings = build_from_config(cfg, str(tmp_path))
    assert rag.invoke("What is the capital of Canada?").answer == "Ottawa" and settings["max_inflight"] == 8
    assert cli_main(["check", "--config", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["llm"] == "ResilientLLM" and out["sources"][0]["retriever"] == "ResilientRetriever"
    assert cli_main(["ask", "--config", str(path), "What is the capital of Italy?"]) == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "Rome"
    from fgao.server.config import server_kwargs

    monkeypatch.setenv("MY_KEYS", "a, b")
    assert server_kwargs(settings)["api_keys"] == ["a", "b"]
    monkeypatch.setenv("FGAO_API_KEYS", "k")
    assert server_kwargs({})["api_keys"] == ["k"]  # the default variable always enables auth
    monkeypatch.delenv("FGAO_API_KEYS")
    assert "api_keys" not in server_kwargs({})
    yml = tmp_path / "fgao.yaml"
    pytest.importorskip("yaml")
    import yaml

    yml.write_text(yaml.safe_dump(cfg))
    from fgao.server import load_config

    assert load_config(str(yml))[0].invoke("What is the capital of France?").answer == "Paris"


def test_request_documents():
    c = client()
    docs = ["Lima is the capital of Peru.", {"text": "Quito is the capital of Ecuador.", "title": "Ecuador"}]
    r = c.post("/v1/answer", json={"question": "What is the capital of Ecuador?", "documents": docs}).json()
    assert r["answer"] == "Quito" and r["support_passage"]["title"] == "Ecuador"
    r = c.post("/v1/answer", json={"question": "What is the capital of Chile?", "documents": docs,
                                   "web_documents": ["Santiago is the capital of Chile."]}).json()
    assert r["answer"] == "Santiago"
    r = c.post("/v1/batch", json={"questions": ["What is the capital of Peru?"], "documents": docs}).json()
    assert r["results"][0]["answer"] == "Lima"
    with c.stream("POST", "/v1/answer/stream", json={"question": "What is the capital of Peru?", "documents": docs}) as s:
        assert '"answer": "Lima"' in "".join(s.iter_text())
    assert c.post("/v1/answer", json={"question": "What is the capital of France?"}).json()["answer"] == "Paris"
    small = TestClient(create_app(FGAORAG.from_retrievers(KB, Scripted(), web=WEB), max_documents=1))
    assert small.post("/v1/answer", json={"question": "q", "documents": docs}).status_code == 422
