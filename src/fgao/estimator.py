"""Training-free online action-effect estimator (Sections 3 and the final correction).

Equation (3) blends a semantic prior with the empirical effect of comparable past
executions using rho = nu / (nu + n) (eq. 4).  Two refinements are made so that
past executions are "meaningfully comparable" (the correction at the end of the
specification):

1. **Comparability rule.**  ``similarity="exact"`` (default) pools only executions
   with the identical signature (tool + arguments).  ``similarity="type"`` pools by
   tool name, i.e. the original action-type average k(a); it is kept for ablation.
   ``similarity="task"`` uses comparability classes declared by the task
   (``Task.comparability_key``): the "carefully specified similarity rule"
   that the specification allows.

   ``rate(action, "raw", c)`` additionally returns the plain average of raw
   deltas v_i(t+1) - v_i(t), which is the literal empirical term of eq. (12) of
   the derivation; FGAOConfig(estimator="raw") uses it for comparison.

2. **State-conditioning.**  The raw change v_i(t+1) - v_i(t) depends on the current
   value v_i(t): an action cannot add progress to a condition that is already
   satisfied.  We therefore estimate *rates* rather than raw deltas:

       gain rate  g_i = E[ max(dv_i, 0) / (1 - v_i) ]   observed when v_i < 1
       harm rate  h_i = E[ max(-dv_i, 0) / v_i ]        observed when v_i > 0

   and predict  delta_i(a) = (1 - v_i) g_i(a) - v_i h_i(a).  Each rate is blended
   by eq. (3)/(4) with its own comparable-observation count.  For binary
   conditions g_i is the success probability and h_i the probability of undoing
   it, which is exactly what the semantic prior declares.

The same machinery estimates the probability that an action makes an observable
predicate true (used by the lookahead) and the expected execution cost.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, Mapping, Tuple

from .types import Action

_EPS = 1e-12


@dataclass
class _Running:
    n: int = 0
    total: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


@dataclass
class _KeyStats:
    executions: int = 0
    cost: _Running = field(default_factory=_Running)
    gain: Dict[str, _Running] = field(default_factory=lambda: defaultdict(_Running))
    harm: Dict[str, _Running] = field(default_factory=lambda: defaultdict(_Running))
    produce: Dict[str, _Running] = field(default_factory=lambda: defaultdict(_Running))
    raw: Dict[str, _Running] = field(default_factory=lambda: defaultdict(_Running))


class EffectStatistics:
    """Online record of executed actions keyed by the comparability rule."""

    def __init__(self, similarity: str = "exact", key_fn=None) -> None:
        if similarity not in ("exact", "type", "task"):
            raise ValueError("similarity must be 'exact', 'type' or 'task'")
        if similarity == "task" and key_fn is None:
            raise ValueError("similarity='task' needs the task's comparability_key")
        self.similarity = similarity
        self.key_fn = key_fn
        self._stats: Dict[Hashable, _KeyStats] = defaultdict(_KeyStats)

    # ----------------------------------------------------------------- keys
    def key(self, action: Action) -> Hashable:
        if self.similarity == "task":
            return self.key_fn(action)  # declared comparability class
        return action.signature if self.similarity == "exact" else action.tool

    # --------------------------------------------------------------- update
    def update(
        self,
        action: Action,
        v_before: Mapping[str, float],
        v_after: Mapping[str, float],
        pred_before: Mapping[str, bool],
        pred_after: Mapping[str, bool],
        cost: float,
    ) -> None:
        s = self._stats[self.key(action)]
        s.executions += 1
        s.cost.add(float(cost))
        for c, vb in v_before.items():
            va = v_after.get(c, vb)
            d = va - vb
            s.raw[c].add(d)  # raw observed delta, eq. (19), for the literal eq. (12)
            if vb < 1.0 - _EPS:
                s.gain[c].add(max(d, 0.0) / (1.0 - vb))
            if vb > _EPS:
                s.harm[c].add(max(-d, 0.0) / vb)
        for p, pb in pred_before.items():
            if not pb:
                s.produce[p].add(1.0 if pred_after.get(p, False) else 0.0)

    def add_gain_observation(self, action: Action, cond: str, v_before: float, v_after: float) -> None:
        """Delayed credit: the outcome of ``action`` on ``cond`` is only revealed by a later check."""
        if v_before < 1.0 - _EPS:
            self._stats[self.key(action)].gain[cond].add(max(v_after - v_before, 0.0) / (1.0 - v_before))

    # -------------------------------------------------------------- queries
    def executions(self, action: Action) -> int:
        k = self.key(action)
        return self._stats[k].executions if k in self._stats else 0

    def _get(self, action: Action):
        return self._stats.get(self.key(action))

    def rate(self, action: Action, kind: str, target: str) -> Tuple[int, float]:
        s = self._get(action)
        if s is None:
            return 0, 0.0
        table = {"gain": s.gain, "harm": s.harm, "produce": s.produce, "raw": s.raw}[kind]
        r = table.get(target)
        if r is None:
            return 0, 0.0
        return r.n, r.mean

    def observed_targets(self, action: Action, kind: str):
        s = self._get(action)
        if s is None:
            return ()
        table = {"gain": s.gain, "harm": s.harm, "produce": s.produce, "raw": s.raw}[kind]
        if kind == "raw":
            return tuple(c for c, r in table.items() if r.n and r.total != 0)
        return tuple(c for c, r in table.items() if r.n and r.total > 0)

    def mean_cost(self, action: Action) -> Tuple[int, float]:
        s = self._get(action)
        if s is None:
            return 0, 0.0
        return s.cost.n, s.cost.mean


def blend(prior: float, empirical: float, n: int, nu: float) -> float:
    """Eq. (3) with eq. (4): rho * prior + (1 - rho) * empirical, rho = nu/(nu+n).

    nu = inf (or feedback disabled) returns the prior; nu = 0 with n > 0 returns
    the empirical mean.
    """
    if n <= 0:
        return prior
    if nu == float("inf"):
        return prior
    rho = nu / (nu + n)
    return rho * prior + (1.0 - rho) * empirical
