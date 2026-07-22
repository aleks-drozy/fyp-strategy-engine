# Phase 4 — Walk-Forward Parameter Tuning — Design Spec

**Date:** 2026-07-13
**Owner:** Aleksandrs Drozdovs
**Status:** approved (brainstorming), pending spec review
**Program:** `fyp-strategy-engine` — Phase 4 of 4

## Purpose

Phase 2 showed the IFVG+CISD strategy, run with the Pine **default** parameters, over-fires (~4×) and
underperforms the real "optimised" logs — implying the edge is largely **parameter selectivity**. Phase 4
tests that hypothesis **honestly**: does a tuned parameter set robustly beat the default **out-of-sample**,
or is any "improvement" just curve-fit hindsight? The method is **walk-forward optimization** — tune on a
past window, measure on the next unseen window, roll forward — so every reported number is out-of-sample.

The honest outcome is **either** "tuning robustly helps OOS" (a real, defensible improvement) **or** "no
setting robustly beats default OOS" (a null — tuning is hindsight). Both are valid, portfolio-grade results.
Success is the rigor of the method, not a pretty number.

## Core principle (the anti-overfitting contract)

- **The headline is the stitched OUT-OF-SAMPLE performance only.** Parameters chosen on a training window are
  applied UNCHANGED to the immediately following test window they never saw; the test-window results are
  concatenated into one continuous OOS record. In-sample numbers are shown only to expose the in-sample→OOS
  gap (the degree of overfitting), never as a headline.
- **The tuning objective is the strategy's OWN OOS profit factor over price data — NOT agreement with the
  real logs.** Tuning to reproduce the winning log would be the curve-fitting trap; the real logs remain a
  sanity reference, never the optimization target.
- **Multiple testing is disclosed.** Report the grid size (combos evaluated per fold); a coarse grid (~100-200
  combos) is used deliberately so we are not silently running thousands of hypotheses.
- **Parameter stability is a reported diagnostic.** If the winning params jump around wildly fold-to-fold, that
  instability is itself evidence of overfitting and is reported as such.

## Data note (carried from Phase 2)

`load_nq()` is a back-adjusted continuous series. This is **fine for Phase 4** because the whole comparison is
strategy-vs-strategy on the **same** series — the per-day offset is intraday-invariant, so signals, outcomes,
and relative $-PnL comparisons between parameter sets are all valid. (Absolute $ figures remain "on the
adjusted series," stated once.) The real logs are used only as an out-of-band sanity reference.

## Walk-forward scheme

Rolling train/test over the usable span (2023-01 → 2025-12; 2023 seeds the first training window):

| Fold | Train (in-sample, tune here) | Test (out-of-sample, measure here) |
|---|---|---|
| 1 | 2023-01 … 2023-12 | 2024-01 … 2024-06 |
| 2 | 2023-07 … 2024-06 | 2024-07 … 2024-12 |
| 3 | 2024-01 … 2024-12 | 2025-01 … 2025-06 |
| 4 | 2024-07 … 2025-06 | 2025-07 … 2025-12 |

- **Rolling 12-month train / 6-month test / 6-month step** → 4 OOS folds covering **2024-01 … 2025-12**
  (2 continuous years out-of-sample). Windows are exact-date parameters, not hard-coded magic.
- **No leakage:** each test window is strictly later than its training window; indicators/signals for a window
  are computed only from bars ≤ that window (the engine is already causal). Warm-up history before a window is
  allowed (it's past data), but selection uses only in-sample trades.

## Parameter grid (coarse, deliberate)

Swept Pine inputs (defaults in **bold**):

- `fvg_threshold` (%): **0**, 0.02, 0.05, 0.10  — filters small gaps (the likely over-firing lever)
- `rr` (risk:reward): 1.0, **1.5**, 2.0, 3.0
- `ema_length`: 10, **20**, 50
- `swing_lookback`: 5, **8**, 12

Base grid = 4×4×3×3 = **144 combos**. Session window is held at the Pine default 09:30–10:30 for the base run
(optionally a 3-value session sweep as a follow-up, gated on runtime). Report the exact combo count used.

**Compute strategy (feasibility):** the full backtest is ~20 s. 144 combos × 4 train windows would be slow if
naive. Split the parameters into a **signal layer** `(fvg_threshold, ema_length, session)` and an **execution
layer** `(rr, swing_lookback)`: signals/indicators are computed **once per signal-layer combo per window** and
reused across all execution-layer combos (which only affect stop/target/exit, not which bars signal). This
cuts indicator recomputation by ~12× (the `rr`×`swing` inner grid). If still slow, run the sweep in the
background. Report total runtime.

## Objective & guards

- **Selection metric (in-sample):** profit factor, subject to a **minimum in-sample trade count `MIN_IS_TRADES = 50`**
  (raised from 20 after review — PF on ~20 trades under a 144-way max is noise-dominated); ties broken by higher
  trade count then lower max drawdown. A documented constant.
- **PRE-REGISTRATION FREEZE (added after adversarial review):** the grid, fold windows, objective, and floor are
  frozen as constants and hashed (SHA-256 + git SHA) into `phase4_results.json`; the runner is single-shot and
  the OOS number is observed once. Changing any of them after seeing an OOS result requires a new dated spec.
- **PRE-REGISTERED FALSIFIABLE SUCCESS RULE + null control:** a positive verdict requires ALL of — tuned
  stitched-OOS PF **> 1.0**; tuned − default OOS PF **≥ 0.10**; tuned beats default in **≥ 3/4 folds**; and
  tuned OOS PF **> the median OOS PF of all 144 combos** (the selection-luck null). All 144 combos' OOS PF are
  recorded so the selected pick's OOS percentile is reported. Otherwise the result is a **null**.
- **OOS metrics reported per fold and stitched:** profit factor, win rate, total PnL, max drawdown, trade
  count, the selected params, the selection-luck null distribution, and the pre-registered verdict.

## Comparisons (what the writeup answers)

1. **Walk-forward tuned (OOS)** vs **default params (same OOS folds)** — does tuning beat default out-of-sample?
2. **In-sample best** vs **its own OOS** per fold — the overfitting gap (how much the in-sample edge decays OOS).
3. **Parameter stability** across folds — do the chosen params agree, or scatter?
4. **Sanity vs real logs** — do the tuned OOS trade counts move toward the real logs' selectivity (fewer,
   better trades)? Reference only, not a target.

## Components (in `fyp-strategy-engine`)

```
fyp-strategy-engine/
  strategy/ifvg.py          # add fvg_threshold param (currently hard-0); keep default 0.0
  strategy/params.py        # StrategyParams dataclass (fvg_threshold, rr, ema_length, swing_lookback, session)
  backtest/engine.py        # accept StrategyParams (thread rr, swing_lookback, ema_length, session, fvg_threshold);
                            #   MUST reproduce Phase-2 results exactly when given defaults (regression-locked)
  tuning/
    grid.py                 # build the coarse parameter grid
    walkforward.py          # fold construction + per-fold tune(in-sample)->select->apply(OOS); no leakage
  run_phase4.py             # run walk-forward -> phase4_results.json + charts
  notebooks/04_parameter_tuning.ipynb
  WRITEUP_PHASE4.md
  tests/  (test_params, test_grid, test_walkforward_no_leakage, test_engine_default_regression, ...)
```

## Testing (TDD)

- **Default-regression:** the parameterized engine, given default `StrategyParams`, reproduces Phase-2's exact
  output (same 605 trades / same aggregate) — proves parameterization didn't change behavior.
- **fvg_threshold:** a higher threshold produces **fewer or equal** FVGs/signals than 0 on the same data.
- **Walk-forward no-leakage:** fold construction yields test windows strictly after train windows; selection on
  a fold uses only in-sample trades (unit test on synthetic trade sets).
- **Selection:** given synthetic in-sample results, the optimizer picks the max-PF combo respecting the trade
  floor and tie-breaks.
- **Grid:** produces the expected combo count; defaults are included in the grid.

## Non-Goals

- **Not** Phase 3 (large-sample Monte Carlo / ML) — that runs AFTER, on the tuned trades if tuning helps.
- **No** new indicators or strategy mechanics — only the existing Pine inputs are swept.
- **No** intra-window parameter changes; one param set per fold.
- **No** tuning to match the real logs (that is the trap this whole design avoids).

## Risks

- **Compute cost** → coarse grid + signal-layer caching + optional background run; report runtime.
- **Multiple testing / overfitting** → OOS-only headline, disclosed grid size, parameter-stability diagnostic.
- **Possible null** → the strategy may be unprofitable OOS under every setting; that is reported honestly as
  the finding, exactly as the Monte Carlo and ML studies reported theirs.
- **Single objective luck** → 4 folds is modest; the parameter-stability and default-comparison diagnostics
  guard against reading too much into one fold.
