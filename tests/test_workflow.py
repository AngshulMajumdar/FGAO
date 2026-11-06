import asyncio

import pytest

from fgao import FGAOConfig, Status, Step, ToolResult, Workflow, WorkflowResult


def build(fail_first=0, on_error="raise"):
    wf = Workflow("etl")
    wf.add_goal("loaded", lambda s: "rows" in s, weight=1)
    wf.add_goal("clean", lambda s: s.get("clean", False), weight=1)
    wf.add_predicate("has_path", lambda s: "path" in s)
    calls = {"n": 0}

    @wf.tool(gain={"loaded": 0.9}, requires=["has_path"], cost=1)
    def load(state):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise IOError("flaky")
        return {"rows": [1, 2, None]}

    @wf.tool(gain={"clean": 0.9}, requires_goals=["loaded"], cost=1)
    def dropna(state):
        return ToolResult(update={"rows": [r for r in state["rows"] if r is not None], "clean": True}, cost=0.5)

    return wf.compile(budget=10, on_error=on_error)


def test_invoke_orders_tools_by_preconditions():
    r = build().invoke({"path": "x.csv"})
    assert r.status == Status.SUCCESS and r.success
    assert [s.tool for s in r.steps] == ["load", "dropna"]
    assert r.state["rows"] == [1, 2] and r.cost == pytest.approx(1.5)
    assert r.trace()[1]["cost"] == 0.5


def test_blocked_without_precondition():
    r = build().invoke({})
    assert r.status == Status.BLOCKED and not r.success and r.steps == []


def test_stream_yields_steps_then_result():
    items = list(build().stream({"path": "x"}))
    assert all(isinstance(i, Step) for i in items[:-1]) and isinstance(items[-1], WorkflowResult)
    seen = []
    build().invoke({"path": "x"}, callbacks=[seen.append])
    assert len(seen) == 2


def test_on_error_record_retries_then_succeeds():
    r = build(fail_first=1, on_error="record").invoke({"path": "x"})
    assert r.success and r.steps[0].error.startswith("OSError") and [s.tool for s in r.steps][:2] == ["load", "load"]
    with pytest.raises(IOError):
        build(fail_first=1).invoke({"path": "x"})


def test_batch_and_async():
    app = build()
    rs = app.batch([{"path": "a"}, {}, {"path": "b"}], max_concurrency=3)
    assert [r.status for r in rs] == [Status.SUCCESS, Status.BLOCKED, Status.SUCCESS]
    rs = asyncio.run(app.abatch([{"path": "a"}, {"path": "b"}]))
    assert all(r.success for r in rs)

    async def collect():
        return [x async for x in app.astream({"path": "a"})]

    items = asyncio.run(collect())
    assert isinstance(items[-1], WorkflowResult) and len(items) == 3


def test_validation_errors():
    wf = Workflow()
    with pytest.raises(ValueError):
        wf.compile()
    wf.add_goal("g", lambda s: 0.0)
    wf.add_tool("t", lambda s: None, gain={"nope": 0.5})
    with pytest.raises(ValueError, match="unknown goal"):
        wf.compile()
    wf = Workflow().add_goal("g", lambda s: 0.0)
    wf.add_tool("t", lambda s: None, requires=["p"])
    with pytest.raises(ValueError, match="unknown predicate"):
        wf.compile()
    wf = Workflow().add_goal("g", lambda s: 0.0)
    wf.add_tool("t", lambda s: None, gain={"g": 1.5})
    with pytest.raises(ValueError, match="outside"):
        wf.compile()
    wf = Workflow().add_goal("g", lambda s: 0.0)
    wf.add_tool("t", lambda s: None, args={"k": 1})
    with pytest.raises(ValueError, match="already declared"):
        wf.add_tool("t", lambda s: None, args={"k": 1})
    with pytest.raises(ValueError):
        wf.add_goal("g", lambda s: 0.0)


def test_bad_check_and_return_values():
    wf = Workflow().add_goal("g", lambda s: 2.0)
    wf.add_tool("t", lambda s: None, gain={"g": 0.5})
    with pytest.raises(ValueError, match="expected a value"):
        wf.compile().invoke({})
    wf = Workflow().add_goal("g", lambda s: 0.0)
    wf.add_tool("t", lambda s: 42, gain={"g": 0.5})
    with pytest.raises(TypeError):
        wf.compile().invoke({})


def test_variants_with_args_and_custom_state():
    class S:
        hits = 0

    wf = Workflow().add_goal("found", lambda s: s.hits > 0)

    def search(s, engine):
        if engine == "good":
            s.hits += 1

    wf.add_tool("search", search, args={"engine": "bad"}, gain={"found": 0.9}, cost=1, group="search")
    wf.add_tool("search", search, args={"engine": "good"}, gain={"found": 0.5}, cost=1, group="search")
    r = wf.compile(FGAOConfig(similarity="exact"), budget=10).invoke(S())
    assert r.success and [str(s.action) for s in r.steps][-1] == "search(engine=good)"


def test_controller_from_workflow():
    app = build()
    c = app.controller(budget=3)
    c.reset(app.observe({"path": "x"}))
    assert c.select_action().action.tool == "load"
