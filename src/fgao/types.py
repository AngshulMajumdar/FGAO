"""Core value types shared by the controller, the workflow API and the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Hashable, Iterable, List, Mapping, Optional, Tuple

SAT_THRESHOLD = 1.0 - 1e-9  # a goal counts as verified when its value is >= this


@dataclass(frozen=True)
class Condition:
    """A verifiable goal condition with weight ``weight``.

    Mandatory conditions must all be verified for the run to succeed.
    """

    name: str
    weight: float = 1.0
    mandatory: bool = True
    description: str = ""


@dataclass(frozen=True, eq=False)
class Action:
    """A concrete operation: a tool name plus fixed arguments.

    ``signature`` is the exact identity used for exact comparability, ``tool``
    the action type. Equality and hash are those of ``(tool, args)``; the hash
    is computed once, since actions are dictionary keys on the hot path.
    """

    tool: str
    args: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.tool, self.args)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self._hash == other._hash and self.tool == other.tool and self.args == other.args  # type: ignore[attr-defined]

    @classmethod
    def of(cls, tool: str, args: Optional[Mapping[str, Any]] = None) -> "Action":
        return cls(tool, tuple((args or {}).items()))

    @property
    def signature(self) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        return (self.tool, self.args)

    def arg(self, key: str, default: Any = None) -> Any:
        for k, val in self.args:
            if k == key:
                return val
        return default

    @property
    def kwargs(self) -> Dict[str, Any]:
        return dict(self.args)

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.args)
        return f"{self.tool}({inner})"


@dataclass(frozen=True)
class Semantics:
    """Declared (prior) semantics of an action.

    gain[c]     declared fraction of the remaining deficiency of goal ``c`` one
                execution closes (for a binary goal: its success probability).
    harm[c]     declared fraction of the achieved value of ``c`` one execution
                destroys.
    produces[p] declared probability that predicate ``p`` becomes true.
    requires_predicates / requires_conditions: preconditions for admissibility.
    cost        declared cost per execution.
    """

    gain: Mapping[str, float] = field(default_factory=dict)
    harm: Mapping[str, float] = field(default_factory=dict)
    produces: Mapping[str, float] = field(default_factory=dict)
    requires_predicates: FrozenSet[str] = frozenset()
    requires_conditions: FrozenSet[str] = frozenset()
    cost: float = 1.0


@dataclass(frozen=True)
class ActionSpec:
    """An action, its declared semantics and (optionally) its comparability group.

    Actions in the same ``group`` share online feedback when the controller is
    configured with ``similarity="task"``.
    """

    action: Action
    semantics: Semantics
    group: Optional[Hashable] = None


@dataclass
class Observation:
    """What the controller sees after each step: verified goal values in [0, 1]
    and observable predicates."""

    values: Dict[str, float]
    predicates: Dict[str, bool] = field(default_factory=dict)

    def copy(self) -> "Observation":
        return Observation(dict(self.values), dict(self.predicates))


@dataclass
class ObsState:
    """Internal controller state at step ``step``."""

    v: Dict[str, float]
    predicates: Dict[str, bool]
    budget: float
    step: int = 0

    def copy(self) -> "ObsState":
        return ObsState(dict(self.v), dict(self.predicates), self.budget, self.step)


@dataclass(frozen=True)
class Scores:
    """Numerical justification of one candidate action at one step."""

    Q: float
    rank: float
    exploit: float
    gain: float
    cost: float
    bonus: float
    continuation: float


@dataclass
class Decision:
    """The controller's choice for the current step."""

    step: int
    action: Action
    scores: Scores
    candidates: List[Tuple[Action, Scores]]  # every affordable admissible action, in declaration order
    n_admissible: int
    select_time: float

    def ranked(self) -> List[Tuple[Action, Scores]]:
        return sorted(self.candidates, key=lambda x: -x[1].rank)


def deficiency(conditions: Iterable[Condition], v: Mapping[str, float]) -> float:
    """Observable deficiency  Phi = sum_i w_i (1 - v_i)."""
    return float(sum(c.weight * (1.0 - v.get(c.name, 0.0)) for c in conditions))


def all_mandatory_verified(conditions: Iterable[Condition], v: Mapping[str, float]) -> bool:
    return all(v.get(c.name, 0.0) >= SAT_THRESHOLD for c in conditions if c.mandatory)


class Status:
    """Termination reasons."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"  # every mandatory goal verified
    STALLED = "STALLED"  # no action has positive value (eq. 20)
    BLOCKED = "BLOCKED"  # no admissible action
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # nothing affordable / budget spent
    STEP_LIMIT = "STEP_LIMIT"
