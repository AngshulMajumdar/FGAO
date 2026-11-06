"""Small random tool-use problems for controller regression tests.

Each problem has weighted goals (some optional), solver actions with declared
rates that may overclaim, preparatory actions that only produce a predicate,
goal dependencies, undisclosed side effects and noisy costs. Execution is
seeded, so a run is a deterministic function of (problem seed, run seed,
chosen actions). ``golden_controller.json`` holds the original v2
implementation's decisions on these problems.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


def make_problem(seed: int):
    rng = random.Random(seed)
    m = rng.randint(2, 6)
    goals = [{"name": f"g{i}", "weight": round(rng.uniform(0.5, 2.0), 3), "mandatory": rng.random() < 0.75}
             for i in range(m)]
    if not any(g["mandatory"] for g in goals):
        goals[0]["mandatory"] = True
    preds: List[str] = []
    actions = []
    for i, g in enumerate(goals):
        needs = []
        if rng.random() < 0.4:
            p = f"ctx:{g['name']}"
            preds.append(p)
            actions.append({"tool": "fetch", "args": [["for", g["name"]]], "kind": "prep", "target": p,
                            "true": rng.uniform(0.5, 1.0), "decl": round(rng.uniform(0.7, 0.95), 3),
                            "cost": round(rng.uniform(0.5, 2.0), 3), "requires": [], "requires_goals": [],
                            "harm_true": {}, "harm_decl": {}})
            needs.append(p)
        req_goals = [goals[i - 1]["name"]] if i > 0 and rng.random() < 0.3 else []
        for j in range(rng.randint(1, 3)):
            true = rng.choice((rng.uniform(0.0, 0.2), rng.uniform(0.4, 0.95)))
            decl = round(min(0.95, true + rng.uniform(0.3, 0.6)) if rng.random() < 0.35 else
                         max(0.05, min(0.95, true + rng.uniform(-0.1, 0.1))), 3)
            harm_true, harm_decl = {}, {}
            if m > 1 and rng.random() < 0.25:
                other = rng.choice([x["name"] for x in goals if x["name"] != g["name"]])
                harm_true[other] = rng.uniform(0.2, 0.8)
                if rng.random() < 0.3:
                    harm_decl[other] = round(harm_true[other], 3)
            actions.append({"tool": rng.choice(("python", "search", "llm", "api")),
                            "args": [["goal", g["name"]], ["v", j]], "kind": "solve", "target": g["name"],
                            "true": true, "decl": decl, "cost": round(rng.uniform(0.5, 3.0), 3),
                            "requires": list(needs), "requires_goals": list(req_goals),
                            "harm_true": harm_true, "harm_decl": harm_decl})
    budget = round(rng.uniform(0.6, 2.5) * sum(a["cost"] for a in actions) / max(1, len(goals)) * 2, 3)
    if rng.random() < 0.1:
        budget = float("inf")
    return {"seed": seed, "goals": goals, "predicates": preds, "actions": actions, "budget": budget}


class Env:
    """Hidden world of a problem."""

    def __init__(self, problem, run_seed: int) -> None:
        self.p = problem
        self.rng = random.Random(problem["seed"] * 1_000_003 + run_seed)
        self.sat = {g["name"]: False for g in problem["goals"]}
        self.pred = {p: False for p in problem["predicates"]}
        self.by_key = {(a["tool"], tuple(tuple(x) for x in a["args"])): a for a in problem["actions"]}

    def values(self) -> Dict[str, float]:
        return {k: 1.0 if v else 0.0 for k, v in self.sat.items()}

    def predicates(self) -> Dict[str, bool]:
        return dict(self.pred)

    def execute(self, tool: str, args) -> float:
        a = self.by_key[(tool, tuple(tuple(x) for x in args))]
        if self.rng.random() < a["true"]:
            if a["kind"] == "prep":
                self.pred[a["target"]] = True
            else:
                self.sat[a["target"]] = True
        for c, h in a["harm_true"].items():
            if self.sat[c] and self.rng.random() < h:
                self.sat[c] = False
        return a["cost"] * (1.0 + 0.2 * (self.rng.random() - 0.5))


CONFIGS = {
    "default": {},
    "exact": {"similarity": "exact"},
    "type": {"similarity": "type"},
    "depth2": {"lookahead_depth": 2},
    "eq16": {"lookahead": "eq16", "lam": 0.0, "nu": 2},
    "eq17": {"lookahead": "eq17", "lam": 0.05, "nu": 2},
    "none": {"lookahead": "none"},
    "raw": {"estimator": "raw"},
    "ratio": {"objective": "ratio"},
    "beta": {"beta": 0.01},
    "nofeedback": {"feedback": False},
    "nocorrective": {"corrective": False},
    "optional_mass": {"optional_mass": 0.1},
}


def declared(a) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """(gain, harm, produces) declared for an action dict."""
    if a["kind"] == "prep":
        return {}, {}, {a["target"]: a["decl"]}
    return {a["target"]: a["decl"]}, dict(a["harm_decl"]), {}


def group(a):
    return (a["tool"], a["target"]) if a["kind"] == "solve" else ("fetch",)
