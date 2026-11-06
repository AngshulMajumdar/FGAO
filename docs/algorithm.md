# The FGAO v2 algorithm

This page specifies exactly what `FGAOController` computes. The code is in `src/fgao/controller.py` and `src/fgao/estimator.py`.

## State

The controller observes three things:
- the goal values *vᵢ* ∈ [0, 1], produced by mechanical checks;
- the boolean predicates;
- the remaining budget *B*.

It never reads a tool's self-reported success.

The deficiency is **Φ = Σᵢ wᵢ (1 − vᵢ)**. A goal counts as verified when *vᵢ* ≥ 1 − 10⁻⁹.

## Admissibility

An action is admissible when all its `requires_predicates` are true and all its `requires_conditions` are verified. It is affordable when its estimated cost is at most *B*.

## Effect estimate (state-normalised rates)

For goal *i* and action *a*, there are two rates:
- the **gain rate** *gᵢ(a)*: the fraction of the remaining deficiency of *i* one execution closes;
- the **harm rate** *hᵢ(a)*: the fraction of the achieved value it destroys.

Each rate is a blend of the declared prior and the empirical mean over comparable past executions:

```
r̂ = ρ · r_declared + (1 − ρ) · r_observed,      ρ = ν / (ν + n)
```

Here *n* is the number of comparable observations for that rate. The observations are recorded as follows:
- **Gain:** max(Δvᵢ, 0) / (1 − vᵢ), recorded when *vᵢ* < 1.
- **Harm:** max(−Δvᵢ, 0) / vᵢ, recorded when *vᵢ* > 0.

The predicted change is **δ̂ᵢ(a) = (1 − vᵢ) ĝᵢ(a) − vᵢ ĥᵢ(a)**, and the predicted progress is **Δ̂(a) = Σᵢ wᵢ δ̂ᵢ(a)**.

The probability that an action produces a predicate, and the action's cost, are blended in the same way.

`estimator="raw"` instead blends the predicted delta for the current state with the plain mean of raw observed deltas.

**Comparability** decides which past executions count as comparable:

| `similarity` | executions pooled together |
|---|---|
| `"exact"` | identical tool and arguments |
| `"type"` | same tool name |
| `"task"` | same declared `group`; the default, used by the RAG pipeline to pool KB variants |

## Score

```
S(a | s) = Δ̂(a | s) − λ ĉ(a) / B₀
Q(a)     = S(a | s) + γ · max(0, H(a))     (+ β · bonus, if β > 0)
```

The continuation value *H* depends on `lookahead`:

| `lookahead` | how *H* is computed |
|---|---|
| `"unlock"` (default) | Assume *a* has its predicted effects. Collect the actions that *a* newly makes admissible, each weighted by the probability that *a* unlocks it. *H* = max over them of the probability-weighted [S(b \| s′)]₊. With `corrective=True`, repair actions also count after a predicted harm, credited with the value *a* creates for them. `lookahead_depth` > 1 recurses. |
| `"eq16"` | Deterministic. Take the predicted next state (harm applied in expectation), then *H* = [max_b S(b \| ŝ′)]₊. |
| `"eq17"` | Outcome expectation. Enumerate the independent predicted effects of *a* (at most 2⁶ outcomes), then *H* = E[ max(0, max_b S(b \| s′)) ]. |
| `"none"` | *H* = 0 |

`objective="ratio"` ranks by (Δ̂ + γH) / ĉ instead. It keeps the same stopping rule.

## Choice and stopping

At each step, in this order:

1. If every mandatory goal is verified, stop with `SUCCESS`.
2. If the step limit is reached, stop with `STEP_LIMIT`.
3. If *B* ≤ 0, stop with `BUDGET_EXHAUSTED`.
4. If no action is admissible, stop with `BLOCKED`.
5. If no admissible action is affordable, stop with `BUDGET_EXHAUSTED`.
6. If the best exploitation value (Q without the bonus) is ≤ 0, stop with `STALLED`.
7. Otherwise, execute the action with the highest rank. Exact ties (within 10⁻¹²) are broken by `random.Random(seed · 7919 + 17)`.

After an execution, `update` does the following:
- charges the measured cost (or the declared cost);
- records the gain, harm, produce and cost observations under the action's comparability key;
- adds the action to the set of actions known to raise any goal it actually raised, which the corrective lookahead uses.

## Cost

Each decision evaluates every affordable admissible action. The cost of one evaluation depends on the lookahead form:

| form | cost of one action's evaluation |
|---|---|
| `"unlock"` | O(m), plus the actions it unlocks |
| `"eq16"` | O(m), plus re-scoring the actions affected by the predicted change |
| `"eq17"` | up to 2⁶ such re-scorings |

There are no model calls and nothing to train.

## Parallel rounds

With `select_batch(K)` (or `Workflow.compile(max_parallel=K)`), a round holds up to K concurrent calls.

1. **The first call** is the sequential v2 choice, unchanged.
2. **Each further call** b, picked greedily, must satisfy all of the following:
   - it is admissible in the current state, not yet in the round, not blocked, and affordable within the budget left after the round's earlier calls;
   - it does not conflict with the round, i.e. shares no declared resource;
   - with `independent=True` (the default), it gains, harms or produces nothing that an earlier call in the round gains, harms or produces;
   - it maximises S(b | ŝ) = Δ̂(b | ŝ) − λĉ(b)/B₀, and that value is positive. Here ŝ is the *expected* state after the earlier calls: each moves every goal to v + δ̂.
3. **The round is observed once.** Each changed goal or predicate is credited to the calls that declare it or have observed it before; a change that no call targets is credited to all of them.

With K = 1 this is exactly v2. Calls in one round cannot learn from each other's outcomes. That is the price of the lower wall-clock time.

## Checkpoints

`snapshot()` stores everything the next decision depends on: goal values, predicates, budget, history, the online statistics, the tie-breaking RNG state (taken before any cached decision) and the blocked actions. After `restore()`, the controller makes exactly the decisions the original would have made.
