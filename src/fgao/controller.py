"""FGAO v2 controller (Feedback-Greedy Agentic Orchestration).

A training-free, model-free numerical controller. At every step it picks

    a* = argmax_a  Delta_hat(a) - lam * c_hat(a) / B0 + gamma * H_hat(a)

where Delta_hat is the predicted weighted progress on the verifiable goals
(declared prior blended with online feedback, rho = nu / (nu + n)), c_hat the
estimated cost, and H_hat the one-step continuation value (lookahead). It stops
when every mandatory goal is verified, or when no admissible, affordable action
has positive value (eq. 20), or when the budget is spent.

The controller performs no I/O. It is driven through three calls::

    ctrl = FGAOController(conditions, actions, budget=16)
    ctrl.reset(observation)
    while ctrl.should_stop() is None:
        decision = ctrl.select_action()
        observation, cost = execute(decision.action)   # your code
        ctrl.update(decision.action, observation, cost)

The numerical algorithm (estimator, scoring, lookahead forms, tie-breaking,
stopping rule) is the tested v2 implementation, unchanged. Defaults are the
configuration used for RAG: unlock lookahead, lam = 0.1, nu = 4, gamma = 0.9,
task-declared comparability groups.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .estimator import EffectStatistics, blend
from .types import (
    SAT_THRESHOLD,
    Action,
    ActionSpec,
    Condition,
    Decision,
    Observation,
    ObsState,
    Scores,
    Semantics,
    Status,
    all_mandatory_verified,
    deficiency,
)

MAX_ENUM_EFFECTS = 6  # eq. 17: at most 2^6 outcomes per action; extra effects fixed at their mode


@dataclass
class FGAOConfig:
    """Controller hyperparameters. Defaults are the evaluated v2 RAG configuration."""

    nu: float = 4.0  # prior strength: rho = nu / (nu + n)
    lam: float = 0.1  # cost weight, in goal units per whole budget
    gamma: float = 0.9  # continuation weight
    lookahead: str = "unlock"  # "unlock" | "eq16" | "eq17" | "none"
    lookahead_depth: int = 1  # unlock form only; 0 disables lookahead
    corrective: bool = True  # unlock form: credit repair actions after predicted harm
    similarity: str = "task"  # feedback pooling: "task" (declared groups) | "exact" | "type"
    estimator: str = "rate"  # "rate" (state-normalised gain/harm rates) | "raw"
    feedback: bool = True  # False: priors only (rho = 1)
    objective: str = "additive"  # "additive" | "ratio" (ranking only)
    beta: float = 0.0  # optional exploration bonus weight
    optional_mass: Optional[float] = None  # weight share of optional goals (None: declared weights)
    cost_scale: Optional[float] = None  # B0; default: the initial budget
    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.lookahead not in ("unlock", "eq16", "eq17", "none"):
            raise ValueError("lookahead must be unlock | eq16 | eq17 | none")
        if self.estimator not in ("rate", "raw"):
            raise ValueError("estimator must be rate | raw")
        if self.objective not in ("additive", "ratio"):
            raise ValueError("objective must be additive | ratio")
        if self.similarity not in ("task", "exact", "type"):
            raise ValueError("similarity must be task | exact | type")


class ControllerStopped(RuntimeError):
    """Raised by ``select_action`` once the controller has stopped."""


ActionsArg = Union[Sequence[ActionSpec], Mapping[Action, Semantics]]


class FGAOController:
    """Step-wise FGAO v2 controller for one episode (one query / one task run).

    Parameters
    ----------
    conditions : goal conditions with weights; mandatory ones define success.
    actions    : ``ActionSpec`` list (or ``{Action: Semantics}``) in a fixed
                 order. The order matters only for exact ties, which are broken
                 by a seeded RNG.
    config     : ``FGAOConfig``.
    budget     : execution budget B0 (``inf`` allowed).
    seed       : seeds the tie-breaking RNG.
    comparability_key : optional ``Action -> Hashable``; overrides ``ActionSpec.group``.
    prior      : optional ``Action -> Semantics`` used for the effect prior only
                 (preconditions and declared costs still come from ``actions``).
    """

    def __init__(
        self,
        conditions: Sequence[Condition],
        actions: ActionsArg,
        config: Optional[FGAOConfig] = None,
        *,
        budget: float = float("inf"),
        seed: int = 0,
        comparability_key: Optional[Callable[[Action], Hashable]] = None,
        prior: Optional[Callable[[Action], Semantics]] = None,
    ) -> None:
        self.cfg = config or FGAOConfig()
        if isinstance(actions, Mapping):
            specs = [ActionSpec(a, s) for a, s in actions.items()]
        else:
            specs = list(actions)
        if len({s.action for s in specs}) != len(specs):
            raise ValueError("duplicate action in the action set")
        self.conditions: List[Condition] = list(conditions)
        names = {c.name for c in self.conditions}
        if len(names) != len(self.conditions):
            raise ValueError("duplicate condition name")
        self.weights = self.assign_weights(self.conditions)
        self.candidates: List[Action] = [s.action for s in specs]
        self._req: Dict[Action, Semantics] = {s.action: s.semantics for s in specs}
        self._sem: Dict[Action, Semantics] = {a: (prior(a) if prior else self._req[a]) for a in self.candidates}
        self._needs_pred: Dict[str, List[Action]] = {}
        self._needs_cond: Dict[str, List[Action]] = {}
        for a in self.candidates:
            for p in self._req[a].requires_predicates:
                self._needs_pred.setdefault(p, []).append(a)
            for c in self._req[a].requires_conditions:
                self._needs_cond.setdefault(c, []).append(a)
        self._init_gainers()
        groups = {s.action: s.group for s in specs if s.group is not None}
        key_fn = None
        if self.cfg.similarity == "task":
            key_fn = comparability_key or (lambda a: groups.get(a, a.signature))
        self._key_fn = key_fn
        self.stats = EffectStatistics(self.cfg.similarity, key_fn)
        self._by_key: Dict[Hashable, List[Action]] = {}
        for a in self.candidates:  # actions sharing feedback: one update changes only their estimates
            self._by_key.setdefault(self.stats.key(a), []).append(a)
        self.n_effect_evals = 0
        self.budget0 = float(budget)
        self._cost_scale = self.cfg.cost_scale or (self.budget0 if math.isfinite(self.budget0) else 1.0)
        self._rng = random.Random(seed * 7919 + 17)
        self.seed = seed
        self._started = False
        self._invalidate()

    def _init_gainers(self) -> None:
        self._gainers: Dict[str, set] = {}
        for a in self.candidates:
            for c, g in self._sem[a].gain.items():
                if g > 0:
                    self._gainers.setdefault(c, set()).add(a)

    def _invalidate_key(self, key: Hashable) -> None:
        for a in self._by_key.get(key, ()):
            self._cost_memo.pop(a, None)
            self._support.pop(a, None)
            self._rate_memo.pop(a, None)

    def _invalidate(self) -> None:
        """Drop memoised estimates. They depend only on the statistics, which change only in ``update``."""
        self._rate_memo: Dict[Action, Dict[Tuple[str, str], float]] = {}
        self._cost_memo: Dict[Action, float] = {}
        self._support: Dict[Action, List[Tuple[str, float, float, float]]] = {}

    # ================================================================ step API
    def reset(self, observation: Observation) -> "FGAOController":
        """Start a fresh episode from the initial observation (online statistics start empty, as in v2)."""
        self.stats = EffectStatistics(self.cfg.similarity, self._key_fn)
        self._init_gainers()
        self.n_effect_evals = 0
        self._rng = random.Random(self.seed * 7919 + 17)
        self.v: Dict[str, float] = dict(observation.values)
        self.preds: Dict[str, bool] = dict(observation.predicates)
        self.budget = self.budget0
        self.spent = 0.0
        self.t = 0
        self.history: List[Dict] = []
        self.status: str = Status.RUNNING
        self._pending: Optional[Decision] = None
        self._rng_at_decide = None
        self.blocked: Set[Action] = set()
        self._started = True
        self._invalidate()
        return self

    def block(self, action: Action) -> None:
        """Exclude ``action`` for the rest of the episode (e.g. a human rejected it)."""
        if action not in self._req:
            raise KeyError(f"unknown action {action}")
        self.blocked.add(action)
        if self._pending is not None and self._pending.action == action:
            self._discard_pending()

    def _discard_pending(self) -> None:
        if self._rng_at_decide is not None:
            self._rng.setstate(self._rng_at_decide)
        self._pending = None
        self._rng_at_decide = None

    @property
    def state(self) -> ObsState:
        return ObsState(dict(self.v), dict(self.preds), self.budget, self.t)

    @property
    def success(self) -> bool:
        return all_mandatory_verified(self.conditions, self.v)

    @property
    def deficiency(self) -> float:
        return deficiency(self.conditions, self.v)

    def admissible_actions(self) -> List[Action]:
        st = self.state
        return [a for a in self.candidates if self._admissible(a, st)]

    def _decide(self) -> None:
        if self.status != Status.RUNNING or self._pending is not None:
            return
        state = ObsState(self.v, self.preds, self.budget, self.t)
        if all_mandatory_verified(self.conditions, self.v):
            self.status = Status.SUCCESS
            return
        if self.t >= self.cfg.max_steps:
            self.status = Status.STEP_LIMIT
            return
        if self.budget <= 0:
            self.status = Status.BUDGET_EXHAUSTED
            return
        t_sel = time.perf_counter()
        self._rng_at_decide = self._rng.getstate()
        admissible = [a for a in self.candidates if self._admissible(a, state) and a not in self.blocked]
        if not admissible:
            self.status = Status.BLOCKED
            return
        affordable = [a for a in admissible if self.estimated_cost(a) <= self.budget]
        if not affordable:
            self.status = Status.BUDGET_EXHAUSTED
            return
        self._begin_step(affordable, state)
        scored = [(a, self.score(a, state)) for a in affordable]
        best_exploit = max(s["exploit"] for _, s in scored)
        if best_exploit <= 0:  # eq. (20)
            self.status = Status.STALLED
            return
        top = max(s["rank"] for _, s in scored)
        ties = [(a, s) for a, s in scored if s["rank"] >= top - 1e-12]
        action, sc = ties[self._rng.randrange(len(ties))] if len(ties) > 1 else ties[0]
        self._pending = Decision(
            step=self.t,
            action=action,
            scores=Scores(**sc),
            candidates=[(a, Scores(**s)) for a, s in scored],
            n_admissible=len(admissible),
            select_time=time.perf_counter() - t_sel,
        )

    def should_stop(self) -> Optional[str]:
        """``None`` while running; otherwise the termination status.

        Calling it computes (and caches) the next decision, so it is cheap to
        follow with ``select_action``.
        """
        if not self._started:
            raise RuntimeError("call reset(observation) first")
        self._decide()
        return None if self.status == Status.RUNNING else self.status

    def select_action(self) -> Decision:
        """The action to execute next, with its scores and every alternative."""
        if self.should_stop() is not None:
            raise ControllerStopped(self.status)
        return self._pending  # type: ignore[return-value]

    def update(
        self,
        action: Action,
        observation: Observation,
        cost: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Dict:
        """Record the outcome of executing ``action``.

        ``observation`` holds the goal values and predicates *after* the
        execution, ``cost`` the measured cost (default: the declared cost).
        Updates budget, history and the online effect estimator.
        """
        if not self._started:
            raise RuntimeError("call reset(observation) first")
        if self.status != Status.RUNNING:
            raise ControllerStopped(self.status)
        if action not in self._req:
            raise KeyError(f"unknown action {action}")
        dec = self._pending if (self._pending is not None and self._pending.action == action) else None
        cost = self._req[action].cost if cost is None else float(cost)
        v_before, p_before = dict(self.v), dict(self.preds)
        self.v = dict(observation.values)
        self.preds = dict(observation.predicates)
        self.budget -= cost
        self.spent += cost
        if self.cfg.feedback:
            self.stats.update(action, v_before, self.v, p_before, self.preds, cost)
            self._invalidate_key(self.stats.key(action))
            for c in self.weights:
                if self.v.get(c, 0.0) > v_before.get(c, 0.0):
                    self._gainers.setdefault(c, set()).add(action)
        observed = sum(self.weights[c] * (self.v.get(c, 0.0) - v_before.get(c, 0.0)) for c in self.weights)
        rec = {
            "step": self.t,
            "action": str(action),
            "Q": dec.scores.Q if dec else None,
            "predicted_gain": dec.scores.gain if dec else None,
            "continuation": dec.scores.continuation if dec else None,
            "estimated_cost": dec.scores.cost if dec else None,
            "observed_effect": observed,
            "cost": cost,
            "budget_left": self.budget,
            "deficiency": deficiency(self.conditions, self.v),
            "values": dict(self.v),
            "error": error,
            "n_actions": len(dec.candidates) if dec else None,
            "select_time": dec.select_time if dec else None,
        }
        self.history.append(rec)
        self.t += 1
        self._pending = None
        self._rng_at_decide = None
        return rec

    # ======================================================== parallel calls
    def select_batch(self, max_parallel: int,
                     conflicts: Optional[Callable[[Action, Action], bool]] = None,
                     independent: bool = True) -> List[Decision]:
        """Up to ``max_parallel`` actions to execute concurrently.

        The first is exactly ``select_action()`` (the v2 choice). Further slots are
        filled greedily with actions that are admissible and affordable now, do not
        conflict with the batch, and still have positive net value
        S(b | s_hat) = Delta_hat(b | s_hat) - lam c_hat(b)/B0 in the *expected* state s_hat
        after the actions already chosen (each chosen action moves every goal to
        v + delta_hat). Actions are never repeated within a batch. With
        ``max_parallel=1`` this is ``[select_action()]``.

        independent=True (default) admits only calls whose effects are disjoint from
        the batch: they may not gain, harm or produce anything an earlier member
        gains, harms or produces. Calls on the same goal are then never run at the
        same time (no hedged duplicates, no conflicting writers), and every observed
        change has at most one declared owner, so feedback attribution stays exact.
        independent=False is the plain expected-state rule.
        """
        first = self.select_action()
        batch = [first]
        if max_parallel <= 1:
            return batch
        real = ObsState(self.v, self.preds, self.budget, self.t)
        s_hat = self._expect(first.action, real.copy())
        left = self.budget - self.estimated_cost(first.action)
        chosen = {first.action}
        touched = set(self._attribution(first.action)) if independent else set()
        t_sel = time.perf_counter()
        while len(batch) < max_parallel:
            best, best_val = None, 0.0
            for a in self.candidates:
                if a in chosen or a in self.blocked or not self._admissible(a, real):
                    continue
                c = self.estimated_cost(a)
                if c > left or (conflicts is not None and any(conflicts(a, b.action) for b in batch)):
                    continue
                if independent and self._attribution(a) & touched:
                    continue
                val = self._net(a, s_hat)
                if val > best_val + 1e-12:
                    best, best_val = a, val
            if best is None:
                break
            g = self.predicted_gain(best, s_hat)
            c = self.estimated_cost(best)
            sc = Scores(Q=best_val, rank=best_val, exploit=best_val, gain=g, cost=c, bonus=0.0, continuation=0.0)
            batch.append(Decision(step=self.t + len(batch), action=best, scores=sc, candidates=[], n_admissible=0,
                                  select_time=0.0))
            chosen.add(best)
            if independent:
                touched |= self._attribution(best)
            left -= c
            s_hat = self._expect(best, s_hat)
        if len(batch) > 1:
            dt = time.perf_counter() - t_sel
            for d in batch[1:]:
                d.select_time = dt / (len(batch) - 1)
        self._batch_decs = {d.action: d for d in batch}
        return batch

    def _expect(self, a: Action, s: ObsState) -> ObsState:
        """Expected state after ``a``: v_i <- v_i + delta_hat_i(a | s)."""
        d = self.predicted_effect(a, s)
        for c, x in d.items():
            s.v[c] = min(1.0, max(0.0, s.v.get(c, 0.0) + x))
        return s

    def _attribution(self, a: Action) -> Set[str]:
        sem = self._sem[a]
        out = {c for c, x in sem.gain.items() if x != 0} | {c for c, x in sem.harm.items() if x != 0}
        out |= set(sem.produces)
        out |= set(self.stats.observed_targets(a, "gain")) | set(self.stats.observed_targets(a, "harm"))
        out |= set(self.stats.observed_targets(a, "produce"))
        return out

    def update_batch(self, actions: Sequence[Action], observation: Observation,
                     costs: Optional[Sequence[Optional[float]]] = None,
                     errors: Optional[Sequence[Optional[str]]] = None) -> List[Dict]:
        """Record a batch executed concurrently, observed once afterwards.

        Feedback attribution: a goal or predicate that changed is credited to the batch
        members that declare it or have observed it before; a change no member
        targets is credited to all of them. A single-action batch is exactly ``update``.
        """
        actions = list(actions)
        costs = list(costs) if costs is not None else [None] * len(actions)
        errors = list(errors) if errors is not None else [None] * len(actions)
        if len(actions) == 1:
            return [self.update(actions[0], observation, costs[0], errors[0])]
        if not self._started:
            raise RuntimeError("call reset(observation) first")
        if self.status != Status.RUNNING:
            raise ControllerStopped(self.status)
        if len(set(actions)) != len(actions):
            raise ValueError("an action appears twice in the batch")
        for a in actions:
            if a not in self._req:
                raise KeyError(f"unknown action {a}")
        decs = dict(getattr(self, "_batch_decs", {}) or {})
        if self._pending is not None:
            decs[self._pending.action] = self._pending
        v0, p0 = dict(self.v), dict(self.preds)
        v1, p1 = dict(observation.values), dict(observation.predicates)
        changed = {c for c in set(v0) | set(v1) if v1.get(c, 0.0) != v0.get(c, 0.0)}
        changed |= {q for q in set(p0) | set(p1) if bool(p1.get(q, False)) != bool(p0.get(q, False))}
        attr = {a: self._attribution(a) for a in actions}
        orphan = {k for k in changed if not any(k in attr[a] for a in actions)}
        recs = []
        rnd = self.t
        for a, cost, err in zip(actions, costs, errors):
            mine = (attr[a] & changed) | orphan
            va = {c: (v1.get(c, 0.0) if c in mine else v0.get(c, 0.0)) for c in set(v0) | set(v1)}
            pa = {q: (p1.get(q, False) if q in mine else p0.get(q, False)) for q in set(p0) | set(p1)}
            cost = self._req[a].cost if cost is None else float(cost)
            if self.cfg.feedback:
                self.stats.update(a, v0, va, p0, pa, cost)
                self._invalidate_key(self.stats.key(a))
                for c in self.weights:
                    if va.get(c, 0.0) > v0.get(c, 0.0):
                        self._gainers.setdefault(c, set()).add(a)
            self.budget -= cost
            self.spent += cost
            dec = decs.get(a)
            recs.append({
                "step": self.t, "round": rnd, "action": str(a), "Q": dec.scores.Q if dec else None,
                "predicted_gain": dec.scores.gain if dec else None,
                "continuation": dec.scores.continuation if dec else None,
                "estimated_cost": dec.scores.cost if dec else None,
                "observed_effect": sum(self.weights[c] * (va.get(c, 0.0) - v0.get(c, 0.0)) for c in self.weights),
                "cost": cost, "budget_left": self.budget, "deficiency": deficiency(self.conditions, v1),
                "values": dict(v1), "error": err, "n_actions": None, "select_time": None,
            })
            self.t += 1
        self.v, self.preds = v1, p1
        self.history.extend(recs)
        self._pending = None
        self._rng_at_decide = None
        self._batch_decs = {}
        return recs

    # ======================================================= checkpointing
    def snapshot(self) -> Dict:
        """JSON-serialisable episode state: goal values, predicates, budget, history,
        online statistics, tie-breaking RNG and blocked actions.

        ``restore`` on a controller built with the same conditions, actions and
        configuration continues with exactly the decisions the original would
        have made.
        """
        if not self._started:
            raise RuntimeError("call reset(observation) first")
        index = {a: i for i, a in enumerate(self.candidates)}
        rep: Dict[Hashable, int] = {}
        for a in self.candidates:
            rep.setdefault(self.stats.key(a), index[a])
        stats = []
        for key, ks in self.stats._stats.items():
            if key not in rep:
                continue
            stats.append({
                "action": rep[key], "executions": ks.executions, "cost": [ks.cost.n, ks.cost.total],
                **{kind: {t: [r.n, r.total] for t, r in getattr(ks, kind).items()}
                   for kind in ("gain", "harm", "produce", "raw")},
            })
        rng = self._rng_at_decide if self._rng_at_decide is not None else self._rng.getstate()
        return {
            "format": 1, "n_actions": len(self.candidates), "conditions": [c.name for c in self.conditions],
            "v": dict(self.v), "predicates": dict(self.preds), "budget": self.budget, "budget0": self.budget0,
            "spent": self.spent, "t": self.t, "status": self.status, "history": list(self.history),
            "stats": stats, "gainers": {c: sorted(index[a] for a in acts) for c, acts in self._gainers.items()},
            "rng": [rng[0], list(rng[1]), rng[2]], "blocked": sorted(index[a] for a in self.blocked),
            "n_effect_evals": self.n_effect_evals,
        }

    def restore(self, snap: Mapping) -> "FGAOController":
        """Load a ``snapshot`` (see there)."""
        from .estimator import _KeyStats, _Running

        if snap.get("format") != 1:
            raise ValueError("unknown snapshot format")
        if snap["n_actions"] != len(self.candidates) or snap["conditions"] != [c.name for c in self.conditions]:
            raise ValueError("snapshot belongs to a different action set or goal set")
        self.reset(Observation(dict(snap["v"]), dict(snap["predicates"])))
        self.budget, self.budget0, self.spent = snap["budget"], snap["budget0"], snap["spent"]
        self.t, self.status, self.history = snap["t"], snap["status"], list(snap["history"])
        self.stats._stats.clear()
        for e in snap["stats"]:
            ks = _KeyStats()
            ks.executions = e["executions"]
            ks.cost = _Running(*e["cost"])
            for kind in ("gain", "harm", "produce", "raw"):
                table = getattr(ks, kind)
                for t, (n, total) in e[kind].items():
                    table[t] = _Running(n, total)
            self.stats._stats[self.stats.key(self.candidates[e["action"]])] = ks
        self._gainers = {c: {self.candidates[i] for i in idx} for c, idx in snap["gainers"].items()}
        r = snap["rng"]
        self._rng.setstate((r[0], tuple(r[1]), r[2]))
        self.blocked = {self.candidates[i] for i in snap["blocked"]}
        self.n_effect_evals = snap.get("n_effect_evals", 0)
        self._invalidate()
        return self

    # ============================================================ v2 algorithm
    def assign_weights(self, conditions) -> Dict[str, float]:
        """assign_weights(conditions).

        Default: the task's weights. With ``optional_mass = eps``, mandatory
        conditions share 1 - eps (in proportion to their task weights) and
        optional conditions share eps.
        """
        w = {c.name: c.weight for c in conditions}
        eps = self.cfg.optional_mass
        if eps is None:
            return w
        mand = [c.name for c in conditions if c.mandatory]
        opt = [c.name for c in conditions if not c.mandatory]
        if not opt:
            return w
        sm = sum(w[c] for c in mand) or 1.0
        so = sum(w[c] for c in opt) or 1.0
        out = {c: (1.0 - eps) * w[c] / sm for c in mand}
        out.update({c: eps * w[c] / so for c in opt})
        return out

    def _admissible(self, a: Action, state: ObsState) -> bool:
        req = self._req[a]
        for p in req.requires_predicates:
            if not state.predicates.get(p, False):
                return False
        for c in req.requires_conditions:
            if state.v.get(c, 0.0) < SAT_THRESHOLD:
                return False
        return True

    # -------------------------------------------------------------- estimates
    def _nu(self) -> float:
        return self.cfg.nu if self.cfg.feedback else float("inf")

    def _rate(self, a: Action, kind: str, target: str) -> float:
        memo = self._rate_memo.get(a)
        if memo is None:
            memo = self._rate_memo[a] = {}
        key = (kind, target)
        r = memo.get(key)
        if r is None:
            r = memo[key] = self._rate_uncached(a, kind, target)
        return r

    def _rate_uncached(self, a: Action, kind: str, target: str) -> float:
        sem = self._sem[a]
        prior = {"gain": sem.gain, "harm": sem.harm, "produce": sem.produces}[kind].get(target, 0.0)
        if not self.cfg.feedback:
            return prior
        n, emp = self.stats.rate(a, kind, target)
        return blend(prior, emp, n, self._nu())

    def predicted_effect(self, a: Action, state: ObsState) -> Dict[str, float]:
        """delta_hat_i(a) for every condition (eqs. 10-12)."""
        self.n_effect_evals += 1
        out = {}
        if self.cfg.estimator == "raw":
            sem = self._sem[a]
            n = self.stats.executions(a) if self.cfg.feedback else 0
            for c in self.conditions:
                vi = state.v.get(c.name, 0.0)
                prior = (1.0 - vi) * sem.gain.get(c.name, 0.0) - vi * sem.harm.get(c.name, 0.0)
                emp = self.stats.rate(a, "raw", c.name)[1] if n else 0.0
                d = blend(prior, emp, n, self._nu())
                out[c.name] = max(-vi, min(1.0 - vi, d))  # keep v + delta in [0, 1]
            return out
        for c in self.conditions:
            vi = state.v.get(c.name, 0.0)
            g = self._rate(a, "gain", c.name)
            h = self._rate(a, "harm", c.name)
            out[c.name] = (1.0 - vi) * g - vi * h
        return out

    def predicted_gain(self, a: Action, state: ObsState) -> float:
        """Delta_hat(a | s) = sum_i w_i delta_hat_i(a), eq. (13).

        For the rate estimator, a goal contributes an exact 0.0 unless the action
        declares it or has a positive observation for it, so only that support is
        summed, in condition order; the result is bit-identical to the full sum.
        """
        if self.cfg.estimator != "rate":
            d = self.predicted_effect(a, state)
            return sum(self.weights[c] * d[c] for c in d)
        self.n_effect_evals += 1
        sup = self._support.get(a)
        if sup is None:
            sup = self._support[a] = self._build_support(a)
        v = state.v
        total = 0.0
        for c, w, g, h in sup:
            vi = v.get(c, 0.0)
            total += w * ((1.0 - vi) * g - vi * h)
        return total

    def _build_support(self, a: Action) -> List[Tuple[str, float, float, float]]:
        sem = self._sem[a]
        keys = {c for c, x in sem.gain.items() if x != 0} | {c for c, x in sem.harm.items() if x != 0}
        if self.cfg.feedback:
            keys |= set(self.stats.observed_targets(a, "gain")) | set(self.stats.observed_targets(a, "harm"))
        out = []
        for cond in self.conditions:
            c = cond.name
            if c in keys:
                out.append((c, self.weights[c], self._rate(a, "gain", c), self._rate(a, "harm", c)))
        return out

    def estimated_cost(self, a: Action) -> float:
        r = self._cost_memo.get(a)
        if r is None:
            r = self._cost_memo[a] = self._estimated_cost_uncached(a)
        return r

    def _estimated_cost_uncached(self, a: Action) -> float:
        prior = self._sem[a].cost
        if not self.cfg.feedback:
            return prior
        n, emp = self.stats.mean_cost(a)
        return blend(prior, emp, n, self._nu())

    def uncertainty(self, a: Action, t: int) -> float:
        n = self.stats.executions(a)
        return math.sqrt(2.0 * math.log(t + 2) / (n + 1))

    def _net(self, b: Action, s: ObsState) -> float:
        """S(b | s) = Delta_hat(b | s) - lam c_hat(b)   (eq. 14)."""
        return self.predicted_gain(b, s) - self.cfg.lam * self.estimated_cost(b) / self._cost_scale

    # ----------------------------------------------- predicted effects of a call
    def _targets(self, a: Action, kind: str) -> Set[str]:
        sem = self._sem[a]
        decl = {"gain": sem.gain, "harm": sem.harm, "produce": sem.produces}[kind]
        out = {k for k, x in decl.items() if x > 0}
        if self.cfg.feedback:
            out |= set(self.stats.observed_targets(a, kind))
            if self.cfg.estimator == "raw" and kind in ("gain", "harm"):
                for c in self.stats.observed_targets(a, "raw"):
                    m = self.stats.rate(a, "raw", c)[1]
                    if (m > 0) == (kind == "gain"):
                        out.add(c)
        return out

    def _effect_probs(self, a: Action, state: ObsState):
        """Independent predicted effects of ``a``: list of (kind, key, prob)."""
        effects = []
        gains = set()
        for c in self._targets(a, "gain"):
            vi = state.v.get(c, 0.0)
            if vi < SAT_THRESHOLD:
                if self.cfg.estimator == "raw":
                    d = self.predicted_effect_single(a, state, c)
                    g = max(0.0, d) / (1.0 - vi)
                else:
                    g = self._rate(a, "gain", c)
                if g > 0:
                    effects.append(("gain", c, min(1.0, g)))
                    gains.add(c)
        for p in self._targets(a, "produce"):
            if not state.predicates.get(p, False):
                q = self._rate(a, "produce", p)
                if q > 0:
                    effects.append(("produce", p, min(1.0, q)))
        for c in self._targets(a, "harm"):
            vi = state.v.get(c, 0.0)
            if vi > 0 and c not in gains:
                if self.cfg.estimator == "raw":
                    d = self.predicted_effect_single(a, state, c)
                    h = max(0.0, -d) / vi
                else:
                    h = self._rate(a, "harm", c)
                if h > 0:
                    effects.append(("harm", c, min(1.0, h)))
        return effects

    def predicted_effect_single(self, a: Action, state: ObsState, c: str) -> float:
        return self.predicted_effect(a, state)[c]

    @staticmethod
    def _apply(state: ObsState, effects, realised, harm_expected: bool = False) -> ObsState:
        s2 = state.copy()
        for (kind, key, prob), r in zip(effects, realised):
            if kind == "gain" and r:
                s2.v[key] = 1.0
            elif kind == "produce" and r:
                s2.predicates[key] = True
            elif kind == "harm":
                if harm_expected:
                    s2.v[key] = state.v.get(key, 0.0) * (1.0 - prob)
                elif r:
                    s2.v[key] = 0.0
        return s2

    # ----------------------------------------- continuation: eq. (15)-(17)
    def _dependents(self, changed_conds: Iterable[str], changed_preds: Iterable[str]) -> Set[Action]:
        """Calls whose admissibility or predicted value can change with these keys."""
        out: Set[Action] = set()
        for c in changed_conds:
            out.update(self._needs_cond.get(c, ()))
            out.update(self._deps.get(c, ()))
        for p in changed_preds:
            out.update(self._needs_pred.get(p, ()))
        return out

    def _best_after(self, s2: ObsState, changed_conds, changed_preds) -> float:
        """max{0, max_{b in A(s2), affordable} S(b | s2)}, reusing S(b | z_t) for unaffected b."""
        affected = self._dependents(changed_conds, changed_preds)
        best = 0.0
        for val, b in self._S_sorted:  # descending
            if val <= best:
                break
            if b in affected:
                continue
            if self._cost[b] <= s2.budget:
                best = val
                break
        for b in affected:
            if self._cost.get(b, self.estimated_cost(b)) > s2.budget or not self._admissible(b, s2):
                continue
            val = self._net(b, s2)
            if val > best:
                best = val
        return best

    def _cont_eq16(self, a: Action, state: ObsState) -> float:
        effects = self._effect_probs(a, state)
        s2 = self._apply(state, effects, [True] * len(effects), harm_expected=True)
        s2.budget = state.budget - self._cost[a]
        changed_c = [k for kind, k, _ in effects if kind in ("gain", "harm")]
        changed_p = [k for kind, k, _ in effects if kind == "produce"]
        return self._best_after(s2, changed_c, changed_p)

    def _cont_eq17(self, a: Action, state: ObsState) -> float:
        effects = self._effect_probs(a, state)
        if len(effects) > MAX_ENUM_EFFECTS:  # keep the most uncertain, fix the rest at their mode
            effects.sort(key=lambda e: abs(e[2] - 0.5))
            fixed, effects = effects[MAX_ENUM_EFFECTS:], effects[:MAX_ENUM_EFFECTS]
            state = self._apply(state, fixed, [e[2] >= 0.5 for e in fixed])
        budget2 = state.budget - self._cost[a]
        total = 0.0
        for realised in itertools.product((True, False), repeat=len(effects)):
            prob = 1.0
            for (kind, key, q), r in zip(effects, realised):
                prob *= q if r else (1.0 - q)
            if prob <= 0:
                continue
            s2 = self._apply(state, effects, realised)
            s2.budget = budget2
            changed_c = [k for (kind, k, _), r in zip(effects, realised) if r and kind in ("gain", "harm")]
            changed_p = [k for (kind, k, _), r in zip(effects, realised) if r and kind == "produce"]
            total += prob * self._best_after(s2, changed_c, changed_p)
        return total

    # unlock-form (earlier specification, eq. 7)
    def _success_state(self, a: Action, state: ObsState):
        s2 = state.copy()
        p_pred: Dict[str, float] = {}
        p_cond: Dict[str, float] = {}
        harmed: List[str] = []
        for kind, key, q in self._effect_probs(a, state):
            if kind == "produce":
                s2.predicates[key] = True
                p_pred[key] = q
            elif kind == "gain":
                s2.v[key] = 1.0
                p_cond[key] = q
            elif kind == "harm" and self.cfg.corrective:
                s2.v[key] = state.v.get(key, 0.0) * (1.0 - q)
                harmed.append(key)
        s2.budget = state.budget - self.estimated_cost(a)
        s2.step = state.step + 1
        return s2, p_pred, p_cond, harmed

    def continuation(self, a: Action, state: ObsState, depth: int) -> float:
        """Unlock form: max_{b in Next(a)} P(unlock b | a) [S(b | a)]_+ (+ repairs)."""
        if depth <= 0:
            return 0.0
        s2, p_pred, p_cond, harmed = self._success_state(a, state)
        if not p_pred and not p_cond and not harmed:
            return 0.0
        unlock = set()
        for p in p_pred:
            unlock.update(self._needs_pred.get(p, ()))
        for c in p_cond:
            unlock.update(self._needs_cond.get(c, ()))
        best = 0.0
        for b in unlock:
            if b == a or self._admissible(b, state) or not self._admissible(b, s2):
                continue
            if self.estimated_cost(b) > s2.budget:
                continue
            req = self._req[b]
            prob = 1.0
            for p in req.requires_predicates:
                if p in p_pred:
                    prob *= p_pred[p]
            for c in req.requires_conditions:
                if c in p_cond:
                    prob *= p_cond[c]
            val = self._net(b, s2)
            if depth > 1:
                val += self.cfg.gamma * max(0.0, self.continuation(b, s2, depth - 1))
            best = max(best, prob * max(0.0, val))
        if harmed:
            repairs = set()
            for c in harmed:
                repairs.update(self._gainers.get(c, ()))
            for b in repairs:
                if b == a or not self._admissible(b, s2) or self.estimated_cost(b) > s2.budget:
                    continue
                now = max(0.0, self._net(b, state)) if self._admissible(b, state) else 0.0
                best = max(best, max(0.0, self._net(b, s2)) - now)
        return best

    # -------------------------------------------------------------- scoring
    def _begin_step(self, actions: List[Action], state: ObsState) -> None:
        """Per-step caches: costs, S(b | z_t), and which conditions each b depends on."""
        self._cost = {a: self.estimated_cost(a) for a in self.candidates}
        self._S = {a: self._net(a, state) for a in actions}
        self._S_sorted = sorted(((v, a) for a, v in self._S.items()), key=lambda x: -x[0])
        deps: Dict[str, Set[Action]] = {}
        if self.cfg.lookahead in ("eq16", "eq17"):
            for b in self.candidates:
                for c in self._targets(b, "gain") | self._targets(b, "harm"):
                    deps.setdefault(c, set()).add(b)
                if self.cfg.estimator == "raw":
                    sem = self._sem[b]
                    for c in set(sem.gain) | set(sem.harm):
                        deps.setdefault(c, set()).add(b)
        self._deps = deps

    def score(self, a: Action, state: ObsState) -> Dict[str, float]:
        gain = self.predicted_gain(a, state)
        cost = self._cost[a] if hasattr(self, "_cost") and a in self._cost else self.estimated_cost(a)
        immediate = gain - self.cfg.lam * cost / self._cost_scale  # S_t(a), eq. (14)
        la = self.cfg.lookahead
        if la == "none" or (la == "unlock" and self.cfg.lookahead_depth <= 0):
            cont = 0.0
        elif la == "unlock":
            cont = self.continuation(a, state, self.cfg.lookahead_depth)
        elif la == "eq16":
            cont = self._cont_eq16(a, state)
        else:
            cont = self._cont_eq17(a, state)
        exploit = immediate + self.cfg.gamma * max(0.0, cont)  # eq. (16)/(18)
        unc = self.uncertainty(a, state.step) if self.cfg.beta > 0 else 0.0
        q = exploit + self.cfg.beta * unc
        if self.cfg.objective == "ratio":
            rank = (gain + self.cfg.gamma * max(0.0, cont)) / max(cost, 1e-9)
        else:
            rank = q
        return {"Q": q, "rank": rank, "exploit": exploit, "gain": gain, "cost": cost,
                "bonus": unc, "continuation": cont}

