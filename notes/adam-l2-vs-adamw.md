# Adam + (λ/N)·L2 vs AdamW for ENN world models

**Status:** idea / partial evidence (2026-10-15). Confirmed for the categorical ENN head on
RiverSwim (commit `f2d0216`). Not tested for the deterministic or Normal heads, which still
default to λ=0. The ENN optimizer is still `optax.adamw(lr, weight_decay=1e-4)` in
`models/enn.py`.

## Claim

For the ENN dynamics models, plain Adam with an explicit L2 penalty in the loss,

    loss = mean_i ℓ_i + (λ / N) · ‖θ_trainable‖²      (N = dataset size, prior params excluded)

may be better than AdamW's decoupled weight decay. With the explicit penalty, epistemic
uncertainty shrinks with the number of visits to an input. AdamW's decay can't produce that
behavior.

## Why

- **L2 in the loss changes the objective.** `Σ_i ℓ_i + λ‖θ‖²` (the mean loss above times
  N) is a MAP objective. Adam's per-parameter preconditioning changes the optimization path
  but not the stationary points, so training settles where the data's pull balances the
  prior. k samples at an input are weighed against a fixed prior strength λ:
  - epistemic uncertainty falls as the visit count rises;
  - the result doesn't depend on how long the model trains.
- **AdamW's decay acts outside the gradient.** Its update is
  `θ ← θ − lr·m/√v − lr·wd·θ`. The data gradient is divided by its own running magnitude
  √v first, so any persistent pull becomes a step of about `lr`, whether it comes from 1
  sample or 1000. The equilibrium is `m/√v ≈ wd·θ`, so weights end up near `1/wd`
  however much evidence supports them. It can't make uncertainty depend on visit counts
  (Loshchilov & Hutter, *Decoupled Weight Decay Regularization*: under Adam, decoupled
  decay is not equivalent to L2).
- **At current settings AdamW's decay is negligible anyway.** The per-step shrink is
  `lr·wd = 1e-3 · 1e-4 = 1e-7`. A sweep up to `wd = 0.1` did not stop the collapse below.

## Evidence (RiverSwim, categorical ENN, prior scale 10)

MI at (state 5, right) after k visits, as a fraction of its unvisited value:

| | k=1 | k=3 | k=10 | k=30 |
|---|---|---|---|---|
| Dirichlet(1) posterior (reference) | 0.87 | 0.68 | 0.39 | 0.18 |
| λ=0, main.py-like training | 0.07 | 0.05 | 0.03 | 0.003 |
| λ=1, main.py-like training | 0.95 | 0.75 | 0.50 | 0.26 |

End-to-end (500 steps/rollout × 10):
- visits to (5, right): 1 → 82
- final model P(stay | 5, right): 1.0 → 0.13–0.24 (true 0.3)

## To do

- Test the (λ/N)·L2 penalty (`_l2_penalty` in `models/enn.py`) on the deterministic and
  Normal heads. Check the visit-count MI curve and validation MAE.
- If it holds up, consider dropping AdamW's weight decay in favor of the penalty.
