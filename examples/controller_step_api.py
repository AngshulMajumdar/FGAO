"""Drive the controller yourself (no Workflow): useful inside an existing agent loop.

    python controller_step_api.py
"""
from fgao import Action, ActionSpec, Condition, FGAOController, Observation, Semantics

world = {"ctx": False, "solved": False}
attempts = {"solve": 0}

actions = [
    ActionSpec(Action("fetch_context"), Semantics(produces={"ctx": 0.9}, cost=1.0)),
    ActionSpec(Action("solve"), Semantics(gain={"solved": 0.8}, requires_predicates=frozenset({"ctx"}), cost=2.0)),
]
ctrl = FGAOController([Condition("solved", weight=1.0)], actions, budget=10)
ctrl.reset(Observation({"solved": 0.0}, {"ctx": False}))

while ctrl.should_stop() is None:
    decision = ctrl.select_action()
    print("chose", decision.action, f"Q={decision.scores.Q:.3f}",
          "alternatives:", [(str(a), round(s.Q, 3)) for a, s in decision.ranked()])
    if decision.action.tool == "fetch_context":
        world["ctx"] = True
    else:
        attempts["solve"] += 1
        world["solved"] = attempts["solve"] >= 2  # first attempt fails
    ctrl.update(decision.action, Observation({"solved": float(world["solved"])}, {"ctx": world["ctx"]}))

print("stopped:", ctrl.status, "spent:", ctrl.spent)
