"""Parallel tool calls: FGAO picks up to K concurrent calls per round."""
import asyncio
import json
import os
import threading
import time

import pytest

import random_tasks as RT
from fgao import Action, ActionSpec, Condition, FGAOConfig, FGAOController, Observation, Semantics, Status, Workflow
from test_golden import GOLD_CTRL, specs_for


def fanout(n_goals=4, delay=0.2, same_goal_alt=False, resources=False):
    """n independent goals, one tool each; optional second tool on goal 0."""
    wf = Workflow("fanout")
    active = {"now": 0, "max": 0}
    lock = threading.Lock()
    for i in range(n_goals):
        wf.add_goal(f"g{i}", (lambda k: lambda s: k in s["done"])(f"g{i}"))

    def work(s, i):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(delay)
        with lock:
            active["now"] -= 1
        s["done"].add(f"g{i}")

    for i in range(n_goals):
        wf.add_tool("work", work, args={"i": i}, gain={f"g{i}": 0.9}, cost=1,
                    resources=("db",) if resources else ())
    if same_goal_alt:
        wf.add_tool("work_alt", lambda s: s["done"].add("g0"), gain={"g0": 0.8}, cost=1)
    return wf, active


def test_parallel_runs_concurrently_and_succeeds():
    wf, active = fanout(4, 0.2)
    t0 = time.perf_counter()
    r = wf.compile(budget=20, max_parallel=4).invoke({"done": set()})
    dt = time.perf_counter() - t0
    assert r.success and len(r.steps) == 4 and active["max"] == 4 and dt < 0.6  # one round, not 0.8 s
    wf, active = fanout(4, 0.05)
    r1 = wf.compile(budget=20).invoke({"done": set()})
    assert r1.success and active["max"] == 1  # max_parallel=1 is sequential


def test_async_parallel():
    wf, active = fanout(4, 0.0)

    async def go():
        return await wf.compile(budget=20, max_parallel=4).ainvoke({"done": set()})

    r = asyncio.run(go())
    assert r.success and len(r.steps) == 4
    rounds = {s.step for s in r.steps}
    assert len(rounds) == 4  # steps are numbered per call


def test_independence_never_batches_same_goal():
    wf, _ = fanout(2, 0.0, same_goal_alt=True)
    c = wf.compile(budget=20, max_parallel=4).controller(budget=20)
    c.reset(wf.compile(budget=20).observe({"done": set()}))
    batch = c.select_batch(4)
    goals = [next(iter(c._sem[d.action].gain)) for d in batch]
    assert len(goals) == len(set(goals))  # disjoint targets
    naive = wf.compile(budget=20).controller(budget=20)
    naive.reset(wf.compile(budget=20).observe({"done": set()}))
    assert len(naive.select_batch(4, independent=False)) == 3  # the plain rule hedges on g0


def test_resources_serialise_tools():
    wf, active = fanout(3, 0.02, resources=True)
    r = wf.compile(budget=20, max_parallel=4).invoke({"done": set()})
    assert r.success and active["max"] == 1


def test_first_slot_is_the_v2_choice():
    for g in [g for g in GOLD_CTRL if g["config"] == "default"][:60]:
        p = RT.make_problem(g["seed"])
        conds = [Condition(x["name"], x["weight"], x["mandatory"]) for x in p["goals"]]
        env = RT.Env(p, g["run_seed"])
        a = FGAOController(conds, specs_for(p), FGAOConfig(), budget=p["budget"], seed=g["run_seed"])
        b = FGAOController(conds, specs_for(p), FGAOConfig(), budget=p["budget"], seed=g["run_seed"])
        obs = Observation(env.values(), env.predicates())
        a.reset(obs)
        b.reset(obs)
        if a.should_stop() is None:
            assert b.select_batch(4)[0].action == a.select_action().action
            assert b.select_batch(1) == [b.select_action()]


def test_batch_attribution():
    A, B = Action("a"), Action("b")
    specs = [ActionSpec(A, Semantics(gain={"x": 0.9})), ActionSpec(B, Semantics(gain={"y": 0.9}))]
    c = FGAOController([Condition("x"), Condition("y")], specs, FGAOConfig(similarity="exact"), budget=10)
    c.reset(Observation({"x": 0.0, "y": 0.0}))
    assert {d.action for d in c.select_batch(2)} == {A, B}  # equal value: v2's seeded tie-break picks the first
    recs = c.update_batch([A, B], Observation({"x": 1.0, "y": 0.0}), [1.0, 1.0])
    assert [r["observed_effect"] for r in recs] == [1.0, 0.0]  # x credited to a only
    assert c.stats.rate(A, "gain", "x")[1] == 1.0 and c.stats.rate(B, "gain", "y") == (1, 0.0)
    assert c.budget == 8 and c.t == 2
    with pytest.raises(ValueError):
        c.update_batch([A, A], Observation({"x": 1.0, "y": 0.0}))


def test_parallel_validation():
    wf, _ = fanout(2, 0.0)
    with pytest.raises(ValueError):
        wf.compile(max_parallel=0)
