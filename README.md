# fyp-strategy-engine

Rebuilding + extending the FYP IFVG+CISD NQ strategy in Python. Phase 1: data foundation.

## Phase 2 — Strategy engine + real-log validation

A faithful, bar-by-bar Python reimplementation of the FYP IFVG+CISD NQ
strategy (session gate, IFVG, CISD, EMA filter, double-confirmation entry,
8-bar swing stop, 1.5R target, 1 trade/day), run with the Pine script's
default parameters over the full Phase-1 dataset and validated against two
real TradingView trade logs.

**Honest headline:** the rebuild recovers 76-80% of the real logs'
trade-days and directions (good recall), but fires ~4x as many trades as
the real logs (precision only 20-25%) and underperforms them on profit
factor and win rate on both a losing and a winning period. It is
directionally consistent on both regimes, which supports the port being
substantively correct — the gap looks like tuning/selectivity in the real
"optimised" track record that the raw default parameters don't capture.
This motivates Phase 4 (parameter sweeps / regime filter). Full result,
interpretation, and data-comparability caveats: see `WRITEUP_STRATEGY.md`.

| | 2023-24 log (losing) | Winning log |
|---|---|---|
| Real baseline (in-window) | 95 trades / −$4,600 | 59 trades / +$18,115 |
| Matched / Missed / Extra | 76 / 19 / 300 | 45 / 14 / 134 |
| Precision / Recall | 0.20 / 0.80 | 0.25 / 0.76 |
| Generated PF / WR | 0.71 / 33.8% | 1.09 / 42.5% |
| Real PF / WR | 0.90 / 37.9% | 1.53 / 55.9% |

**Run it:**

```
.venv/Scripts/python run_backtest.py
```

Requires the Phase-1 raw data at `data/raw/Dataset_NQ_1min_2022_2025.csv`
(not committed) and the two real trade-log CSVs referenced in
`run_backtest.py`. Writes `backtest_results.json` and three charts to
`charts/` (equity curve, coverage bars, generated-vs-real PF/WR). Takes
~20 seconds — the engine is a pure bar-by-bar loop (no lookahead) over
~1.05M 1-minute rows.

See `notebooks/03_strategy_engine.ipynb` for the same run narrated
end-to-end with the coverage tables and charts inline, and
`WRITEUP_STRATEGY.md` for the full writeup.

## Phase 4 — Walk-forward parameter tuning

Parameterized the Phase-2 engine (`fvg_threshold`, `rr`, `ema_length`,
`swing_lookback`) and ran a pre-registered walk-forward optimization: tune
on a rolling 12-month in-sample window (144-combo grid, max profit factor
subject to ≥50 in-sample trades), apply the selected parameters *unchanged*
to the following 6-month out-of-sample window, across 4 rolling folds
spanning 2024-01 through 2025-12. The grid, fold dates, selection floor,
and objective were frozen as constants *before* the run, which is
single-shot and records a config hash + git SHA as an audit trail.

**Honest headline: `robust_improvement = FALSE` — a null result.** The
pre-registered, falsifiable success rule needed all four of: tuned
stitched-OOS PF > 1.0 (**fails**, 0.9945); tuned − default OOS PF margin ≥
0.10 (**fails**, +0.0747); tuned beats default in ≥3/4 folds (passes, 3/4);
tuned OOS PF beats the median-combo selection-luck null (passes, 0.9945 >
0.9396). Two of four fail, so tuning does not robustly clear the bar. It
does, however, **directionally confirm** Phase 2's hypothesis: tuning
nearly erases the default's two-year OOS loss (−$907.50 vs. −$20,362.50)
by trading 42% less often (233 vs. 401 trades) and lifting profit factor
0.92 → 0.99 — selectivity helps substantially, it just doesn't reach
breakeven on this data. Full result, the selection-luck null, parameter
stability (with its n=4/overlapping-window caveat), and every disclosed
caveat: see `WRITEUP_PHASE4.md`.

**Run it:**

```
.venv/Scripts/python run_phase4.py
```

Requires the same Phase-1 raw data as Phase 2. Takes ~10.6 minutes (144
combos × 4 folds × a full null-control OOS pass over all 144 combos per
fold). Writes `phase4_results.json` and five charts to `charts/`
(equity curve, per-fold OOS PF, parameter stability, in-sample-vs-OOS PF,
selection-luck null). See `notebooks/04_parameter_tuning.ipynb` for the
same result narrated with the fold table and all five charts inline
(loads the committed `phase4_results.json` — it does not re-run the
sweep), and `WRITEUP_PHASE4.md` for the full writeup.

## Phase 5 — Costs + exits + volatility filter

Extended the Phase-2/4 engine with a pluggable exit handler (5 modes:
`fixed_1_5R`, `breakeven_1R`, `trail_swing`, `partial_1R`, `time_stop`), a
leak-free ATR%-based volatility gate, and a reason-aware cost model
(commission + slippage, charged per fill), all off/default by default so
Phase-2/4 behavior is byte-preserved (regression-locked against the
Phase-2 golden fixture). Swept `exit_mode × vol_filter` (20 combos, entry
parameters fixed at the Phase-4-null defaults) through the same walk-forward
harness as Phase 4, selecting **net** in-sample PF, and evaluated a
5-condition pre-registered success rule against **net**, cost-adjusted,
stitched out-of-sample results.

**Honest headline: `robust_improvement = FALSE` — a null result, and the
closest this program has come.** For the first time, the tuned
configuration's stitched net OOS profit factor crosses breakeven: PF
**1.0707** on 228 trades, **+$9,795** net P&L, **+$42.96/trade**
expectancy (base: PF 0.9006, **−$25,587.50**, **−$63.81/trade**). It
passes 4 of 5 pre-registered conditions — net PF > 1.0, margin ≥ 0.10 over
base, beats base in 3/4 folds, beats the 75th-percentile selection-luck
null — and fails only the statistical-confidence gate: a bootstrap CI
lower bound of **0.807**, below the 1.0 floor the rule demands. Why: the
entire net edge is concentrated in one 25-trade fold (+$18,845), while the
other three folds combined net −$9,050. A **promising signal, not a
proven edge** — exactly what the pre-registered gate was built to
distinguish. Full result, the Fold-4 concentration analysis, cost
sensitivity (0×/1×/2×), and every mandatory disclosure: see
`WRITEUP_PHASE5.md`.

**Run it:**

```
.venv/Scripts/python run_phase5.py
```

Requires the same Phase-1 raw data as Phase 2/4. Takes ~52 seconds (20
combos × 4 folds). Writes `phase5_results.json` and five charts to
`charts/` (stitched equity curve, per-fold OOS PF, selected exit/filter
stability, cost-sensitivity bars, 20-combo selection-luck null). See
`notebooks/05_costs_exits_volfilter.ipynb` for the same result narrated
with the fold table, eligibility table, cost sensitivity, and all five
charts inline (loads the committed `phase5_results.json` — it does not
re-run the sweep), and `WRITEUP_PHASE5.md` for the full writeup.

## Program complete (Phases 1–5)

Data foundation (P1) → faithful strategy rebuild that located the likely
source of the real track record's edge in selectivity/tuning rather than a
different core mechanic (P2) → a large-sample Monte Carlo/ML re-run (P3)
pre-registered as conditional on Phase 4 finding a robust improvement, and
correctly **not run** since it didn't (`docs/superpowers/specs/2026-07-13-phase4-parameter-tuning-design.md`)
→ a pre-registered, falsifiable walk-forward tuning study that tests the
Phase-2 hypothesis honestly out-of-sample and reports an **honest null**
(P4) → smarter exits + costs + a volatility filter, layered on the
Phase-4-null default entry, which pushes the stitched net OOS point
estimate above breakeven for the first time — but is correctly held back
from being called a proven edge by the same pre-registered statistical-
confidence gate that makes every other result in this program trustworthy
(P5). Every headline number in this program was defined before it was
observed and reported whether or not it flattered the strategy — see
`WRITEUP_PHASE5.md`'s closing "Program epilogue" for the full 5-phase
retrospective.
