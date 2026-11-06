"""Declarative workflow API: goals + tools, orchestrated by the FGAO controller.

LangGraph wires nodes with edges you write by hand. Here you declare *what
must become true* (goals with mechanical checks) and *what each tool claims to
do* (gain, harm, produced predicates, preconditions, cost); FGAO decides the
order at run time from those declarations plus online feedback, and stops
when the goals are verified or nothing left is worth its cost.

    wf = Workflow()
    wf.add_goal("has_data", check=lambda s: "rows" in s, weight=1)
    wf.add_tool("load", load_fn, gain={"has_data": 0.9}, cost=1)
    app = wf.compile(budget=10)
    result = app.invoke({"path": "data.csv"})

State is any object you choose (a dict is the common case). Tools receive the
state and their declared arguments, may mutate it, and may return

    None                     state was mutated in place
    dict                     merged into a dict state (``state.update(d)``)
    ToolResult(update, cost, error)   the same, with a measured cost / error

Checks map the state to a value in [0, 1] (``bool`` is accepted).
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, FrozenSet, Hashable, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from .controller import FGAOConfig, FGAOController
from .types import Action, ActionSpec, Condition, Decision, Observation, Semantics, Status


@dataclass
class ToolResult:
    """Optional rich return value of a tool."""

    update: Optional[Mapping[str, Any]] = None
    cost: Optional[float] = None  # measured cost; default: declared cost
    error: Optional[str] = None


@dataclass
class Tool:
    name: str
    fn: Optional[Callable[..., Any]]  # sync implementation (None for async-only tools)
    action: Action
    semantics: Semantics
    group: Optional[Hashable] = None
    description: str = ""
    afn: Optional[Callable[..., Any]] = None  # async implementation (used by the async API)
    resources: FrozenSet[str] = frozenset()  # tools sharing a resource never run in the same parallel round


@dataclass
class Goal:
    name: str
    check: Callable[[Any], Union[float, bool]]
    weight: float = 1.0
    mandatory: bool = True
    description: str = ""


@dataclass
class Step:
    """One executed step (yielded by ``stream``)."""

    step: int
    action: Action
    decision: Decision
    values_before: Dict[str, float]
    values_after: Dict[str, float]
    cost: float
    budget_left: float
    error: Optional[str] = None

    @property
    def tool(self) -> str:
        return self.action.tool

    def to_dict(self) -> Dict[str, Any]:
        s = self.decision.scores
        return {
            "step": self.step, "action": str(self.action), "Q": s.Q, "predicted_gain": s.gain,
            "continuation": s.continuation, "estimated_cost": s.cost, "cost": self.cost,
            "budget_left": self.budget_left, "values": dict(self.values_after), "error": self.error,
        }


@dataclass
class WorkflowResult:
    """Final outcome of a run (also the last item yielded by ``stream``)."""

    state: Any
    status: str
    success: bool
    values: Dict[str, float]
    cost: float
    steps: List[Step] = field(default_factory=list)

    def trace(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


Callback = Callable[[Step], None]


def _as_value(x: Union[float, bool]) -> float:
    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"goal check returned {x!r}; expected a value in [0, 1]")
    return v


class Workflow:
    """Builder for an FGAO-orchestrated workflow."""

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self.goals: List[Goal] = []
        self.predicates: Dict[str, Callable[[Any], bool]] = {}
        self.tools: List[Tool] = []

    # ------------------------------------------------------------ declaration
    def add_goal(
        self,
        name: str,
        check: Callable[[Any], Union[float, bool]],
        *,
        weight: float = 1.0,
        mandatory: bool = True,
        description: str = "",
    ) -> "Workflow":
        """Declare a verifiable goal. ``check(state)`` must use evidence, not a tool's own claim."""
        if any(g.name == name for g in self.goals):
            raise ValueError(f"goal {name!r} already declared")
        self.goals.append(Goal(name, check, weight, mandatory, description))
        return self

    def add_predicate(self, name: str, fn: Callable[[Any], bool]) -> "Workflow":
        """Declare an observable predicate (used in preconditions and ``produces``)."""
        self.predicates[name] = fn
        return self

    def add_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        args: Optional[Mapping[str, Any]] = None,
        cost: float = 1.0,
        gain: Optional[Mapping[str, float]] = None,
        harm: Optional[Mapping[str, float]] = None,
        produces: Optional[Mapping[str, float]] = None,
        requires: Iterable[str] = (),
        requires_goals: Iterable[str] = (),
        group: Optional[Hashable] = None,
        description: str = "",
        afn: Optional[Callable[..., Any]] = None,
        resources: Iterable[str] = (),
    ) -> Action:
        """Declare one concrete call ``fn(state, **args)`` and its declared effects.

        Register the same function several times with different ``args`` to
        offer several variants (e.g. one per retrieval source). Variants that
        share a ``group`` share online feedback.

        ``fn`` may be an ``async def`` function (async-only tool), and ``afn``
        an optional async twin of a sync ``fn``. The async API awaits async
        implementations and runs sync ones in worker threads.
        """
        if inspect.iscoroutinefunction(fn):
            fn, afn = None, fn
        action = Action.of(name, args)
        if any(t.action == action for t in self.tools):
            raise ValueError(f"tool call {action} already declared")
        sem = Semantics(
            gain=dict(gain or {}), harm=dict(harm or {}), produces=dict(produces or {}),
            requires_predicates=frozenset(requires), requires_conditions=frozenset(requires_goals), cost=float(cost),
        )
        self.tools.append(Tool(name, fn, action, sem, group, description, afn, frozenset(resources)))
        return action

    def add_agent(
        self,
        name: str,
        agent: Any,
        *,
        inputs: Callable[[Any], Any],
        output: Callable[[Any, Any], Optional[Mapping[str, Any]]],
        measured_cost: Union[bool, Callable[[Any], float]] = True,
        budget: Optional[Callable[[Any], Optional[float]]] = None,
        **declared: Any,
    ) -> Action:
        """Register a sub-agent as one tool: anything with ``invoke`` (and optionally ``ainvoke``),
        e.g. another compiled ``Workflow``, an ``FGAORAG``, a LangGraph graph or a LangChain agent.

        inputs(state) -> the sub-agent's input; output(state, result) -> the update for this state.
        measured_cost: True charges ``result.cost`` when the result has one (FGAO sub-workflows and
        FGAORAG do), a function computes it from the result, False charges the declared ``cost``.
        budget(state) -> per-call budget passed to FGAO sub-agents (``budget=`` keyword).
        """
        def _kw(state):
            b = budget(state) if budget else None
            return {"budget": b} if b is not None else {}

        def _wrap(state, res):
            if callable(measured_cost):
                cost = float(measured_cost(res))
            elif measured_cost and hasattr(res, "cost"):
                cost = float(res.cost)
            else:
                cost = None
            return ToolResult(update=output(state, res), cost=cost)

        def fn(state, **_):
            return _wrap(state, agent.invoke(inputs(state), **_kw(state)))

        afn = None
        if hasattr(agent, "ainvoke"):
            async def afn(state, **_):  # noqa: F811
                return _wrap(state, await agent.ainvoke(inputs(state), **_kw(state)))

        return self.add_tool(name, fn, afn=afn, **declared)

    def tool(self, name: Optional[str] = None, **declared: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of ``add_tool``."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add_tool(name or fn.__name__, fn, **declared)
            return fn

        return deco

    def validate(self) -> None:
        goals = {g.name for g in self.goals}
        if not goals:
            raise ValueError("declare at least one goal")
        if not self.tools:
            raise ValueError("declare at least one tool")
        for t in self.tools:
            s = t.semantics
            for c in list(s.gain) + list(s.harm) + list(s.requires_conditions):
                if c not in goals:
                    raise ValueError(f"{t.action}: unknown goal {c!r}")
            for p in list(s.produces) + list(s.requires_predicates):
                if p not in self.predicates:
                    raise ValueError(f"{t.action}: unknown predicate {p!r} (declare it with add_predicate)")
            for d in (s.gain, s.harm, s.produces):
                for k, x in d.items():
                    if not 0.0 <= x <= 1.0:
                        raise ValueError(f"{t.action}: declared rate {k}={x} outside [0, 1]")
            if s.cost < 0:
                raise ValueError(f"{t.action}: negative cost")

    def compile(
        self,
        config: Optional[FGAOConfig] = None,
        *,
        budget: float = float("inf"),
        on_error: str = "raise",
        max_parallel: int = 1,
        independent: bool = True,
    ) -> "CompiledWorkflow":
        """Freeze the declarations into a runnable app.

        on_error="raise" propagates tool exceptions; "record" charges the
        declared cost, records the error and lets feedback learn from it.

        max_parallel > 1 runs up to that many tool calls concurrently per round
        (see ``FGAOController.select_batch``): the first is the v2 choice, further
        slots take calls of positive value in the expected state that, with
        independent=True, touch none of the batch's goals or predicates and share
        no ``resources``. max_parallel=1 is sequential v2.
        """
        if on_error not in ("raise", "record"):
            raise ValueError("on_error must be 'raise' or 'record'")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self.validate()
        return CompiledWorkflow(self, config or FGAOConfig(), budget, on_error, max_parallel, independent)


class CompiledWorkflow:
    """A runnable workflow. Every run gets its own controller, so runs are independent
    and can execute concurrently (``batch`` / ``abatch``) as long as the tools
    themselves are thread-safe."""

    def __init__(self, wf: Workflow, config: FGAOConfig, budget: float, on_error: str, max_parallel: int = 1,
                 independent: bool = True) -> None:
        self.name = wf.name
        self.config = config
        self.budget = budget
        self.on_error = on_error
        self.max_parallel = max_parallel
        self.independent = independent
        self._goals = list(wf.goals)
        self._preds = dict(wf.predicates)
        self._tools = {t.action: t for t in wf.tools}
        self._specs = [ActionSpec(t.action, t.semantics, t.group) for t in wf.tools]
        self.conditions = [Condition(g.name, g.weight, g.mandatory, g.description) for g in self._goals]
        self.async_only = [t.action for t in wf.tools if t.fn is None]

    @property
    def actions(self) -> List[Action]:
        return [s.action for s in self._specs]

    def observe(self, state: Any) -> Observation:
        return Observation(
            {g.name: _as_value(g.check(state)) for g in self._goals},
            {p: bool(fn(state)) for p, fn in self._preds.items()},
        )

    def controller(self, *, budget: Optional[float] = None, seed: int = 0) -> FGAOController:
        """A fresh step-wise controller over this workflow's goals and tools."""
        return FGAOController(self.conditions, self._specs, self.config,
                              budget=self.budget if budget is None else budget, seed=seed)

    # ------------------------------------------------------------- execution
    def _call(self, tool: Tool, state: Any):
        ok, out = self._raw(tool, state)
        if not ok:
            return out
        return self._apply(tool, state, out)

    def _raw(self, tool: Tool, state: Any):
        """(True, output) or (False, (None, error)) without touching the state."""
        if tool.fn is None:
            raise TypeError(f"{tool.action} is async-only; use ainvoke / astream / abatch")
        try:
            return True, tool.fn(state, **tool.action.kwargs)
        except Exception as e:  # noqa: BLE001
            if self.on_error == "raise":
                raise
            return False, (None, f"{type(e).__name__}: {e}")

    async def _araw(self, tool: Tool, state: Any):
        try:
            if tool.afn is not None:
                return True, await tool.afn(state, **tool.action.kwargs)
            return True, await asyncio.to_thread(tool.fn, state, **tool.action.kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if self.on_error == "raise":
                raise
            return False, (None, f"{type(e).__name__}: {e}")

    def _conflicts(self, a: Action, b: Action) -> bool:
        return bool(self._tools[a].resources & self._tools[b].resources)

    def _finish_round(self, ctrl, state, obs, batch, outs, callbacks):
        results = []
        for d, (ok, out) in zip(batch, outs):  # merge updates in batch order
            results.append(self._apply(self._tools[d.action], state, out) if ok else out)
        after = self.observe(state)
        recs = ctrl.update_batch([d.action for d in batch], after, [c for c, _ in results], [e for _, e in results])
        steps = []
        for d, rec, (_, err) in zip(batch, recs, results):
            st = Step(rec["step"], d.action, d, dict(obs.values), dict(after.values), rec["cost"], rec["budget_left"], err)
            steps.append(st)
            for cb in callbacks:
                cb(st)
        return after, steps

    async def _acall(self, tool: Tool, state: Any):
        try:
            if tool.afn is not None:
                out = await tool.afn(state, **tool.action.kwargs)
            else:
                out = await asyncio.to_thread(tool.fn, state, **tool.action.kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if self.on_error == "raise":
                raise
            return None, f"{type(e).__name__}: {e}"
        return self._apply(tool, state, out)

    def _apply(self, tool: Tool, state: Any, out: Any):
        cost, err, upd = None, None, None
        if isinstance(out, ToolResult):
            upd, cost, err = out.update, out.cost, out.error
        elif isinstance(out, Mapping):
            upd = out
        elif out is not None:
            raise TypeError(f"{tool.action} returned {type(out).__name__}; expected None, dict or ToolResult")
        if upd:
            if not isinstance(state, MutableMapping):
                raise TypeError(f"{tool.action} returned an update but the state is not a mutable mapping")
            state.update(upd)
        return cost, err

    def stream(
        self,
        state: Any,
        *,
        seed: int = 0,
        budget: Optional[float] = None,
        callbacks: Sequence[Callback] = (),
    ) -> Iterator[Union[Step, WorkflowResult]]:
        """Run step by step, yielding each executed ``Step`` and finally the ``WorkflowResult``."""
        ctrl = self.controller(budget=budget, seed=seed)
        obs = self.observe(state)
        ctrl.reset(obs)
        steps: List[Step] = []
        if self.max_parallel > 1:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
                while ctrl.should_stop() is None:
                    batch = ctrl.select_batch(self.max_parallel, self._conflicts, self.independent)
                    outs = list(ex.map(lambda d: self._raw(self._tools[d.action], state), batch))
                    obs, new = self._finish_round(ctrl, state, obs, batch, outs, callbacks)
                    steps.extend(new)
                    yield from new
            yield WorkflowResult(state, ctrl.status, ctrl.status == Status.SUCCESS, dict(ctrl.v), ctrl.spent, steps)
            return
        while ctrl.should_stop() is None:
            d = ctrl.select_action()
            tool = self._tools[d.action]
            cost, err = self._call(tool, state)
            after = self.observe(state)
            ctrl.update(d.action, after, cost, err)
            rec = ctrl.history[-1]
            st = Step(d.step, d.action, d, dict(obs.values), dict(after.values), rec["cost"], ctrl.budget, err)
            obs = after
            steps.append(st)
            for cb in callbacks:
                cb(st)
            yield st
        yield WorkflowResult(state, ctrl.status, ctrl.status == Status.SUCCESS, dict(ctrl.v), ctrl.spent, steps)

    def start(self, state: Any, *, seed: int = 0, budget: Optional[float] = None,
              interrupt_before: Iterable[str] = (), callbacks: Sequence[Callback] = ()) -> "Run":
        """A step-by-step run that can pause for approval and be checkpointed (see ``Run``).

        A Run executes one call per step (sequential v2), whatever ``max_parallel`` is.
        """
        return Run(self, state, seed=seed, budget=budget, interrupt_before=interrupt_before, callbacks=callbacks)

    def resume(self, snapshot: Mapping[str, Any], state: Any, *, interrupt_before: Iterable[str] = (),
               callbacks: Sequence[Callback] = ()) -> "Run":
        """Continue a run from ``Run.snapshot()`` with the (restored) application state."""
        run = Run(self, state, seed=snapshot["seed"], budget=snapshot["controller"]["budget0"],
                  interrupt_before=interrupt_before, callbacks=callbacks, _snapshot=snapshot)
        return run

    def invoke(self, state: Any, *, seed: int = 0, budget: Optional[float] = None,
               callbacks: Sequence[Callback] = ()) -> WorkflowResult:
        """Run to termination and return the ``WorkflowResult``."""
        out = None
        for out in self.stream(state, seed=seed, budget=budget, callbacks=callbacks):
            pass
        return out  # type: ignore[return-value]

    def batch(self, states: Sequence[Any], *, max_concurrency: int = 8, seed: int = 0,
              budget: Optional[float] = None) -> List[WorkflowResult]:
        """Run many independent states concurrently in threads; results keep input order."""
        if max_concurrency <= 1:
            return [self.invoke(s, seed=seed, budget=budget) for s in states]
        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            return list(ex.map(lambda s: self.invoke(s, seed=seed, budget=budget), states))

    async def astream(
        self,
        state: Any,
        *,
        seed: int = 0,
        budget: Optional[float] = None,
        callbacks: Sequence[Callback] = (),
    ) -> AsyncIterator[Union[Step, WorkflowResult]]:
        """Native async run: async tools are awaited, sync tools run in worker threads.

        Makes exactly the same decisions as ``stream``; cancelling the task
        cancels the tool call in flight.
        """
        ctrl = self.controller(budget=budget, seed=seed)
        obs = self.observe(state)
        ctrl.reset(obs)
        steps: List[Step] = []
        if self.max_parallel > 1:
            while ctrl.should_stop() is None:
                batch = ctrl.select_batch(self.max_parallel, self._conflicts, self.independent)
                outs = await asyncio.gather(*(self._araw(self._tools[d.action], state) for d in batch))
                obs, new = self._finish_round(ctrl, state, obs, batch, outs, callbacks)
                steps.extend(new)
                for st in new:
                    yield st
            yield WorkflowResult(state, ctrl.status, ctrl.status == Status.SUCCESS, dict(ctrl.v), ctrl.spent, steps)
            return
        while ctrl.should_stop() is None:
            d = ctrl.select_action()
            tool = self._tools[d.action]
            cost, err = await self._acall(tool, state)
            after = self.observe(state)
            ctrl.update(d.action, after, cost, err)
            rec = ctrl.history[-1]
            st = Step(d.step, d.action, d, dict(obs.values), dict(after.values), rec["cost"], ctrl.budget, err)
            obs = after
            steps.append(st)
            for cb in callbacks:
                cb(st)
            yield st
        yield WorkflowResult(state, ctrl.status, ctrl.status == Status.SUCCESS, dict(ctrl.v), ctrl.spent, steps)

    async def ainvoke(self, state: Any, *, seed: int = 0, budget: Optional[float] = None,
                      callbacks: Sequence[Callback] = ()) -> WorkflowResult:
        out = None
        async for out in self.astream(state, seed=seed, budget=budget, callbacks=callbacks):
            pass
        return out  # type: ignore[return-value]

    async def abatch(self, states: Sequence[Any], *, max_concurrency: int = 64, **kw: Any) -> List[WorkflowResult]:
        sem = asyncio.Semaphore(max(1, max_concurrency))

        async def one(s: Any) -> WorkflowResult:
            async with sem:
                return await self.ainvoke(s, **kw)

        return list(await asyncio.gather(*(one(s) for s in states)))


@dataclass
class Interrupt:
    """Returned by ``Run.step`` instead of executing a tool listed in ``interrupt_before``."""

    step: int
    action: Action
    decision: Decision


class Run:
    """A resumable, interruptible workflow run.

        run = app.start(state, interrupt_before=["send_email"])
        out = run.run()                      # runs until done or an Interrupt
        if isinstance(out, Interrupt):
            save(run.snapshot(), state)      # e.g. wait for a human
            run.approve()                    # or run.reject() to exclude that call
            out = run.run()

    ``snapshot()`` is JSON-serialisable; ``app.resume(snapshot, state)`` continues
    with exactly the decisions the uninterrupted run would make. The application
    state itself is yours to persist alongside it.
    """

    def __init__(self, app: CompiledWorkflow, state: Any, *, seed: int, budget: Optional[float],
                 interrupt_before: Iterable[str], callbacks: Sequence[Callback], _snapshot: Optional[Mapping] = None):
        self.app, self.state, self.seed = app, state, seed
        self.interrupt_before = set(interrupt_before)
        self.callbacks = list(callbacks)
        self.ctrl = app.controller(budget=budget, seed=seed)
        self.steps: List[Step] = []
        self.interrupted: Optional[Interrupt] = None
        self._approved: Optional[Action] = None
        self.result: Optional[WorkflowResult] = None
        if _snapshot is None:
            self._obs = app.observe(state)
            self.ctrl.reset(self._obs)
        else:
            self.ctrl.restore(_snapshot["controller"])
            self._obs = Observation(dict(self.ctrl.v), dict(self.ctrl.preds))
            if _snapshot.get("interrupted") is not None:
                d = self.ctrl.select_action()
                if d.action != self.app.actions[_snapshot["interrupted"]]:  # pragma: no cover - guarded by determinism
                    raise RuntimeError("snapshot does not reproduce the interrupted decision")
                self.interrupted = Interrupt(d.step, d.action, d)
            if _snapshot.get("approved") is not None:
                self._approved = self.app.actions[_snapshot["approved"]]

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def status(self) -> str:
        return self.result.status if self.result else (
            "INTERRUPTED" if self.interrupted else self.ctrl.status)

    def _finish(self) -> WorkflowResult:
        self.result = WorkflowResult(self.state, self.ctrl.status, self.ctrl.status == Status.SUCCESS,
                                     dict(self.ctrl.v), self.ctrl.spent, list(self.steps))
        return self.result

    def _next(self):
        """Decision to execute now, or an Interrupt / final result."""
        if self.result is not None:
            return self.result
        if self.interrupted is not None and self._approved is None:
            return self.interrupted
        if self.ctrl.should_stop() is not None:
            return self._finish()
        d = self.ctrl.select_action()
        gated = d.action.tool in self.interrupt_before or str(d.action) in self.interrupt_before
        if gated and self._approved != d.action:
            self.interrupted = Interrupt(d.step, d.action, d)
            return self.interrupted
        return d

    def _record(self, d: Decision, cost, err) -> Step:
        after = self.app.observe(self.state)
        self.ctrl.update(d.action, after, cost, err)
        rec = self.ctrl.history[-1]
        st = Step(d.step, d.action, d, dict(self._obs.values), dict(after.values), rec["cost"], self.ctrl.budget, err)
        self._obs = after
        self.steps.append(st)
        self.interrupted, self._approved = None, None
        for cb in self.callbacks:
            cb(st)
        return st

    def step(self) -> Union[Step, Interrupt, WorkflowResult]:
        """Execute one tool call (sync); returns the Step, an Interrupt, or the final result."""
        d = self._next()
        if not isinstance(d, Decision):
            return d
        cost, err = self.app._call(self.app._tools[d.action], self.state)
        return self._record(d, cost, err)

    async def astep(self) -> Union[Step, Interrupt, WorkflowResult]:
        d = self._next()
        if not isinstance(d, Decision):
            return d
        cost, err = await self.app._acall(self.app._tools[d.action], self.state)
        return self._record(d, cost, err)

    def run(self) -> Union[WorkflowResult, Interrupt]:
        """Step until the run ends or pauses."""
        while True:
            out = self.step()
            if not isinstance(out, Step):
                return out

    async def arun(self) -> Union[WorkflowResult, Interrupt]:
        while True:
            out = await self.astep()
            if not isinstance(out, Step):
                return out

    def approve(self) -> None:
        """Allow the interrupted call; it executes on the next ``step`` / ``run``."""
        if self.interrupted is None:
            raise RuntimeError("nothing to approve")
        self._approved = self.interrupted.action

    def reject(self) -> None:
        """Exclude the interrupted call for the rest of the run; FGAO picks something else."""
        if self.interrupted is None:
            raise RuntimeError("nothing to reject")
        self.ctrl.block(self.interrupted.action)
        self.interrupted, self._approved = None, None

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serialisable checkpoint of the orchestration (not of your application state)."""
        index = {a: i for i, a in enumerate(self.app.actions)}
        return {
            "workflow": self.app.name, "seed": self.seed, "controller": self.ctrl.snapshot(),
            "interrupted": index[self.interrupted.action] if self.interrupted else None,
            "approved": index[self._approved] if self._approved is not None else None,
        }
