import math

import pytest

from fgao import (Action, ActionSpec, Condition, ControllerStopped, FGAOConfig, FGAOController, Observation,
                  Semantics, Status, blend)
from fgao.estimator import EffectStatistics

A, B, PREP = Action("a"), Action("b"), Action.of("prep", {"x": 1})


def ctrl(specs, conds=None, **kw):
    conds = conds or [Condition("g", 1.0)]
    return FGAOController(conds, specs, kw.pop("config", None), **kw)


def obs(g=0.0, **preds):
    return Observation({"g": g}, preds)


def test_blend():
    assert blend(0.8, 0.0, 0, 4) == 0.8
    assert blend(0.8, 0.0, 4, 4) == pytest.approx(0.4)
    assert blend(0.8, 0.2, 3, 0) == pytest.approx(0.2)
    assert blend(0.8, 0.2, 3, math.inf) == 0.8


def test_estimator_state_normalised_rates():
    st = EffectStatistics("exact")
    st.update(A, {"g": 0.5}, {"g": 1.0}, {}, {}, 1.0)
    assert st.rate(A, "gain", "g") == (1, pytest.approx(1.0))
    st.update(A, {"g": 1.0}, {"g": 0.0}, {}, {}, 1.0)
    assert st.rate(A, "harm", "g") == (2, pytest.approx(0.5))  # 0 harm at v=0.5, full harm at v=1


def test_success_immediately():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}))], budget=5).reset(obs(1.0))
    assert c.should_stop() == Status.SUCCESS
    with pytest.raises(ControllerStopped):
        c.select_action()


def test_requires_reset():
    with pytest.raises(RuntimeError):
        ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}))]).should_stop()


def test_picks_higher_value_and_learns():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}, cost=1)), ActionSpec(B, Semantics(gain={"g": 0.6}, cost=1))],
             budget=10).reset(obs())
    d = c.select_action()
    assert d.action == A and len(d.candidates) == 2 and d.scores.gain == pytest.approx(0.9)
    for _ in range(3):  # a keeps failing: feedback pulls its estimate below b
        c.update(A, obs(), 1.0)
        if c.select_action().action == B:
            break
    assert c.select_action().action == B


def test_stalls_when_cost_exceeds_value():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.01}, cost=10))], budget=10,
             config=FGAOConfig(lam=1.0)).reset(obs())
    assert c.should_stop() == Status.STALLED


def test_blocked_and_budget():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}, requires_predicates=frozenset({"p"})))], budget=5).reset(obs())
    assert c.should_stop() == Status.BLOCKED
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}, cost=6))], budget=5).reset(obs())
    assert c.should_stop() == Status.BUDGET_EXHAUSTED


def test_lookahead_takes_preparatory_step():
    specs = [ActionSpec(PREP, Semantics(produces={"ready": 0.95}, cost=1)),
             ActionSpec(A, Semantics(gain={"g": 0.9}, requires_predicates=frozenset({"ready"}), cost=1))]
    c = ctrl(specs, budget=10).reset(obs(ready=False))
    d = c.select_action()
    assert d.action == PREP and d.scores.gain == 0 and d.scores.continuation > 0
    c.update(PREP, obs(ready=True), 1.0)
    assert c.select_action().action == A
    c.update(A, obs(1.0, ready=True), 1.0)
    assert c.should_stop() == Status.SUCCESS and c.success and c.spent == 2.0
    # without lookahead the preparatory call has no value and the controller stalls
    c = ctrl(specs, budget=10, config=FGAOConfig(lookahead="none")).reset(obs(ready=False))
    assert c.should_stop() == Status.STALLED


def test_groups_share_feedback():
    a1, a2 = Action.of("ret", {"src": 1}), Action.of("ret", {"src": 2})
    specs = [ActionSpec(a1, Semantics(gain={"g": 0.8}), group="kb"), ActionSpec(a2, Semantics(gain={"g": 0.8}), group="kb")]
    c = ctrl(specs, budget=10).reset(obs())
    c.update(a1, obs(), 1.0)
    st = c.state
    assert c.predicted_gain(a2, st) == pytest.approx(0.8 * 4 / 5)
    c2 = ctrl(specs, budget=10, config=FGAOConfig(similarity="exact")).reset(obs())
    c2.update(a1, obs(), 1.0)
    assert c2.predicted_gain(a2, c2.state) == pytest.approx(0.8)


def test_update_validation_and_history():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.9}, cost=2))], budget=10).reset(obs())
    with pytest.raises(KeyError):
        c.update(B, obs())
    rec = c.update(A, obs(1.0))  # cost defaults to the declared cost
    assert rec["cost"] == 2 and c.budget == 8 and c.history[0]["action"] == "a()"
    assert c.should_stop() == Status.SUCCESS
    with pytest.raises(ControllerStopped):
        c.update(A, obs(1.0))


def test_step_limit_and_mapping_actions():
    c = FGAOController([Condition("g")], {A: Semantics(gain={"g": 0.9})}, FGAOConfig(max_steps=1), budget=10).reset(obs())
    c.update(c.select_action().action, obs())
    assert c.should_stop() == Status.STEP_LIMIT


def test_config_validation():
    for bad in ({"lookahead": "x"}, {"estimator": "x"}, {"objective": "x"}, {"similarity": "x"}):
        with pytest.raises(ValueError):
            FGAOConfig(**bad)
    with pytest.raises(ValueError):
        FGAOController([Condition("g")], [ActionSpec(A, Semantics()), ActionSpec(A, Semantics())])


def test_ranked_alternatives():
    c = ctrl([ActionSpec(A, Semantics(gain={"g": 0.5})), ActionSpec(B, Semantics(gain={"g": 0.7}))], budget=10).reset(obs())
    d = c.select_action()
    assert [a for a, _ in d.ranked()] == [B, A]
