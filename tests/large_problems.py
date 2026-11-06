"""Large random tool sets (100-1500 tools, up to 60 goals) for scaling regression tests.

``golden_large.json`` holds the decisions of the unoptimised v2 controller on
these problems; the optimised controller must reproduce them exactly.
"""
import random

from fgao import Action, ActionSpec, Condition, Semantics


def make(n_tools, n_goals, seed):
    rng = random.Random(seed)
    goals = [Condition(f"g{i}", round(rng.uniform(.5, 2), 3), mandatory=rng.random() < .8) for i in range(n_goals)]
    if not any(g.mandatory for g in goals):
        goals[0] = Condition(goals[0].name, goals[0].weight, True)
    preds = [f"p{i}" for i in range(max(1, n_goals // 2))]
    specs = []
    for j in range(n_tools):
        if rng.random() < 0.2:
            sem = Semantics(produces={rng.choice(preds): round(rng.uniform(.5, .95), 3)}, cost=round(rng.uniform(.5, 2), 3))
        else:
            g = rng.sample(goals, rng.randint(1, 3))
            req = frozenset({rng.choice(preds)}) if rng.random() < .3 else frozenset()
            rq_c = frozenset({rng.choice(goals).name}) if rng.random() < .1 else frozenset()
            harm = {rng.choice(goals).name: round(rng.uniform(.1, .5), 3)} if rng.random() < .2 else {}
            sem = Semantics(gain={c.name: round(rng.uniform(.1, .9), 3) for c in g}, harm=harm, requires_predicates=req,
                            requires_conditions=rq_c, cost=round(rng.uniform(.5, 3), 3))
        specs.append(ActionSpec(Action.of(f"tool{j % 40}", {"v": j}), sem, group=f"tool{j % 40}"))
    true = {s.action: (rng.random(), rng.random()) for s in specs}  # hidden success / side-effect draws per tool
    return goals, preds, specs, true


def run(ctrl_cls, cfg_cls, n_tools, n_goals, seed, cfg_kw=None, steps=25):
    goals, preds, specs, true = make(n_tools, n_goals, seed)
    rng = random.Random(seed + 1)
    sem = {s.action: s.semantics for s in specs}
    budget = sum(s.semantics.cost for s in specs) / len(specs) * steps * 0.8
    c = ctrl_cls(goals, specs, cfg_cls(**(cfg_kw or {})), budget=budget, seed=seed)
    v = {g.name: 0.0 for g in goals}
    p = {x: False for x in preds}
    c.reset_obs = None
    from fgao import Observation

    c.reset(Observation(dict(v), dict(p)))
    trace = []
    for _ in range(steps):
        if c.should_stop() is not None:
            break
        d = c.select_action()
        s = sem[d.action]
        ok, side = true[d.action]
        for x in s.gain:
            if rng.random() < ok:
                v[x] = 1.0
        for x in s.produces:
            if rng.random() < ok:
                p[x] = True
        for x in s.harm:
            if rng.random() < side:
                v[x] = 0.0
        trace.append((str(d.action), round(d.scores.Q, 10)))
        c.update(d.action, Observation(dict(v), dict(p)), s.cost * (0.8 + 0.4 * rng.random()))
    return {"status": c.status, "trace": trace, "spent": round(c.spent, 10)}


CASES = [(100, 10, 0), (200, 20, 1), (300, 30, 2), (500, 20, 3), (800, 40, 4), (1200, 60, 5), (1500, 30, 6)]
CONFIGS = {"default": {}, "eq16": {"lookahead": "eq16"}, "depth2": {"lookahead_depth": 2}, "exact": {"similarity": "exact"}}
