"""Regression: the library reproduces the original tested FGAO v2 decisions exactly."""
import json
import os

import pytest

import random_tasks as RT
import scenarios as S
from fgao import (Action, ActionSpec, BM25Retriever, Condition, Document, FGAOConfig, FGAOController, FGAORAG,
                  FunctionRetriever, Observation, Semantics, Workflow)

HERE = os.path.dirname(__file__)
GOLD_CTRL = json.load(open(os.path.join(HERE, "golden_controller.json")))
GOLD_RAG = json.load(open(os.path.join(HERE, "golden_rag_v2.json")))


def specs_for(p):
    out = []
    for a in p["actions"]:
        g, h, pr = RT.declared(a)
        sem = Semantics(g, h, pr, frozenset(a["requires"]), frozenset(a["requires_goals"]), a["cost"])
        out.append(ActionSpec(Action(a["tool"], tuple(tuple(x) for x in a["args"])), sem, RT.group(a)))
    return out


def run_controller(p, run_seed, cfg):
    env = RT.Env(p, run_seed)
    conds = [Condition(g["name"], g["weight"], g["mandatory"]) for g in p["goals"]]
    ctrl = FGAOController(conds, specs_for(p), cfg, budget=p["budget"], seed=run_seed)
    ctrl.reset(Observation(env.values(), env.predicates()))
    actions, qs = [], []
    while ctrl.should_stop() is None:
        d = ctrl.select_action()
        cost = env.execute(d.action.tool, d.action.args)
        actions.append(str(d.action))
        qs.append(round(d.scores.Q, 10))
        ctrl.update(d.action, Observation(env.values(), env.predicates()), cost)
    return ctrl.status, actions, qs, round(ctrl.spent, 10), env.values()


@pytest.mark.parametrize("config", sorted(RT.CONFIGS))
def test_controller_matches_original(config):
    cfg = FGAOConfig(**RT.CONFIGS[config])
    cases = [g for g in GOLD_CTRL if g["config"] == config]
    assert cases
    for g in cases:
        got = run_controller(RT.make_problem(g["seed"]), g["run_seed"], cfg)
        assert got == (g["status"], g["actions"], g["Q"], g["cost"], g["final"]), (config, g["seed"], g["run_seed"])


def test_workflow_matches_original():
    """The same problems through the declarative Workflow API (default configuration)."""
    for g in [g for g in GOLD_CTRL if g["config"] == "default"]:
        p = RT.make_problem(g["seed"])
        wf = Workflow()
        for goal in p["goals"]:
            wf.add_goal(goal["name"], (lambda n: lambda env: env.values()[n])(goal["name"]),
                        weight=goal["weight"], mandatory=goal["mandatory"])
        for pr in p["predicates"]:
            wf.add_predicate(pr, (lambda n: lambda env: env.predicates()[n])(pr))
        for spec, a in zip(specs_for(p), p["actions"]):
            gain, harm, prod = RT.declared(a)

            def call(env, _a=a, **kw):
                from fgao import ToolResult
                return ToolResult(cost=env.execute(_a["tool"], _a["args"]))

            wf.add_tool(a["tool"], call, args=dict(spec.action.args), cost=a["cost"], gain=gain, harm=harm,
                        produces=prod, requires=a["requires"], requires_goals=a["requires_goals"], group=spec.group)
        app = wf.compile(budget=p["budget"])
        env = RT.Env(p, g["run_seed"])
        r = app.invoke(env, seed=g["run_seed"])
        assert r.status == g["status"]
        assert [str(s.action) for s in r.steps] == g["actions"]
        assert [round(s.decision.scores.Q, 10) for s in r.steps] == g["Q"]
        assert round(r.cost, 10) == g["cost"]


def _rag_for(sc):
    kb, alld = S.build(sc)
    kb_r = BM25Retriever([Document(d["text"], d["title"], d["id"]) for d in kb])
    all_r = BM25Retriever([Document(d["text"], d["title"], d["id"]) for d in alld])
    web = FunctionRetriever(lambda q, k: [Document(S.web_snippet(d.text), d.title, d.id + "@web")
                                          for d in all_r.retrieve(q, k)])
    return FGAORAG.from_retrievers(kb_r, lambda prompt: S.scripted_reply(prompt, sc["profile"]), web=web,
                                   budget=sc["budget"])


def test_rag_matches_original():
    for g in GOLD_RAG:
        sc = S.make_scenario(g["seed"])
        r = _rag_for(sc).invoke(sc["question"])
        llm_calls = sum(v for k, v in r.calls.items() if not k.startswith("retrieve"))
        assert (r.answer, r.grounded_answer, r.status) == (g["answer"], g["grounded_answer"], g["status"]), g["seed"]
        assert [str(s.action) for s in r.steps] == g["trace"], g["seed"]
        assert [round(s.decision.scores.Q, 10) for s in r.steps] == g["Q"], g["seed"]
        assert r.cost == pytest.approx(g["cost"]) and llm_calls == g["llm_calls"], g["seed"]


def test_rag_async_matches_original():
    """Native async path (awaited operations) makes the same decisions as the original."""
    import asyncio

    async def go():
        for g in GOLD_RAG:
            sc = S.make_scenario(g["seed"])
            r = await _rag_for(sc).ainvoke(sc["question"])
            assert (r.answer, r.status) == (g["answer"], g["status"]), g["seed"]
            assert [str(s.action) for s in r.steps] == g["trace"], g["seed"]
            assert [round(s.decision.scores.Q, 10) for s in r.steps] == g["Q"], g["seed"]

    asyncio.run(go())


def test_rag_async_native_llm_matches_original():
    """Same, with an async-only LLM (awaited directly, never run in a thread)."""
    import asyncio

    async def go():
        for g in GOLD_RAG[:100]:
            sc = S.make_scenario(g["seed"])
            kb, alld = S.build(sc)
            kb_r = BM25Retriever([Document(d["text"], d["title"], d["id"]) for d in kb])
            all_r = BM25Retriever([Document(d["text"], d["title"], d["id"]) for d in alld])

            async def web(q, k):
                return [Document(S.web_snippet(d.text), d.title, d.id + "@web") for d in all_r.retrieve(q, k)]

            async def llm(prompt, _p=sc["profile"]):
                await asyncio.sleep(0)
                return S.scripted_reply(prompt, _p)

            rag = FGAORAG.from_retrievers(kb_r, llm, web=FunctionRetriever(web), budget=sc["budget"])
            r = await rag.ainvoke(sc["question"])
            assert [str(s.action) for s in r.steps] == g["trace"] and r.answer == g["answer"], g["seed"]
            with pytest.raises(TypeError):
                rag.invoke(sc["question"])  # async-only components refuse the sync path

    asyncio.run(go())
