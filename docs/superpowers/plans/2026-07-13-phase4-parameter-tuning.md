# Phase 4 — Walk-Forward Parameter Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Determine — honestly, out-of-sample — whether tuning the strategy's parameters robustly beats the Pine defaults, via walk-forward optimization.

**Architecture:** Parameterize the Phase-2 engine (a `StrategyParams` dataclass threaded through indicators + execution), split it into a cacheable **signal layer** and a cheap **execution layer**, then run a rolling walk-forward: tune on each in-sample window, apply the selected params UNCHANGED to the next out-of-sample window, stitch the OOS results. Report OOS-only headlines + overfitting diagnostics.

**Tech Stack:** Python, pandas 2.2.3, numpy 2.1.3, pytest 8.3.3, matplotlib. Builds on Phase 2's `strategy/` + `backtest/`.

## Global Constraints

- **`StrategyParams` fields + defaults (defaults MUST reproduce Phase 2 exactly):** `fvg_threshold=0.0`, `rr=1.5`, `ema_length=20`, `swing_lookback=8`, `session_start="09:30"`, `session_end="10:30"`.
- **`fvg_threshold` semantics (Pine lines 307-315):** a bullish gap qualifies iff `low[i] > high[i-2]` AND `(low[i]-high[i-2])/high[i-2]*100 >= fvg_threshold`; bearish iff `high[i] < low[i-2]` AND `(low[i-2]-high[i])/low[i-2]*100 >= fvg_threshold`. At `0.0` this is identical to the current gap-exists check (a real positive gap always has pct ≥ 0), so defaults are unchanged.
- **Walk-forward folds (rolling 12mo train / 6mo test / 6mo step):** exact date windows, not magic numbers —
  F1 train 2023-01-01..2023-12-31 / test 2024-01-01..2024-06-30; F2 train 2023-07-01..2024-06-30 / test 2024-07-01..2024-12-31; F3 train 2024-01-01..2024-12-31 / test 2025-01-01..2025-06-30; F4 train 2024-07-01..2025-06-30 / test 2025-07-01..2025-12-11 (clipped to the data edge). OOS span = 2024-01..2025-12.
- **Coarse grid (144 combos):** `fvg_threshold ∈ {0, 0.02, 0.05, 0.10}`, `rr ∈ {1.0, 1.5, 2.0, 3.0}`, `ema_length ∈ {10, 20, 50}`, `swing_lookback ∈ {5, 8, 12}`. Session fixed at default for the base run. Defaults MUST be a grid point.
- **No leakage:** params for a test window are selected using ONLY that fold's in-sample (train-window) trades; each test window is strictly later than its train window. Indicators are causal (a bar's value depends only on bars ≤ it), so precomputing them over the full series and slicing per window is valid — it never uses future bars, only more/less *past* history. Selection never touches test-window data.
- **Objective (in-sample selection):** max **profit factor**, subject to **≥ `MIN_IS_TRADES = 20`** in-sample trades; tie-break higher trade count, then lower max drawdown. If no combo meets the floor in a fold, select the DEFAULT params for that fold (documented fallback) and flag it.
- **Headline = stitched OOS only.** In-sample numbers appear solely to show the in-sample→OOS decay. Report grid size, per-fold selected params, parameter stability, and OOS-tuned vs OOS-default.
- **Data caveat (from Phase 2, restate once):** `load_nq` is back-adjusted; all comparisons here are strategy-vs-strategy on the same series, so signals/outcomes/relative-PnL are valid. Real logs are a sanity reference only, never the tuning target.
- **Reuse:** `metrics.py` (`profit_factor, win_rate, total_pnl, max_drawdown`); Phase-2 `strategy/` + `backtest/`.

---

## Task 1: Parameterize the engine (regression-locked)

**Files:**
- Create: `strategy/params.py`
- Modify: `strategy/ifvg.py` (add `fvg_threshold` param), `backtest/engine.py` (accept `StrategyParams`; split into signal-layer + execution-layer)
- Test: `tests/test_params.py`, `tests/test_ifvg_threshold.py`, `tests/test_engine_default_regression.py`

**Interfaces:**
- Produces: `StrategyParams` dataclass; `compute_ifvg(df, in_session, fvg_threshold=0.0)`; `compute_signal_layer(df, params) -> dict` (keys: `sig` np.ndarray, `ema_v` np.ndarray, `sess` np.ndarray, `o,h,l,c` np.ndarrays, `days` array, `index`); `run_execution(layer, params, fill_mode="next_open") -> list[Trade]`; `backtest(df, params=StrategyParams(), fill_mode="next_open") -> list[Trade]` (= compute_signal_layer + run_execution).

- [ ] **Step 1: Write `strategy/params.py`.**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategyParams:
    fvg_threshold: float = 0.0
    rr: float = 1.5
    ema_length: int = 20
    swing_lookback: int = 8
    session_start: str = "09:30"
    session_end: str = "10:30"
```

- [ ] **Step 2: Add `fvg_threshold` to `compute_ifvg`.** Change the signature to `compute_ifvg(df, in_session, fvg_threshold: float = 0.0)`. Replace the two creation guards:

```python
        bullish_gap = lows[i] > highs[i - 2]
        if bullish_gap and session[i]:
            gap_pct = (lows[i] - highs[i - 2]) / highs[i - 2] * 100.0
            if gap_pct >= fvg_threshold:
                fvg_array.insert(0, { ... })   # unchanged body
        bearish_gap = highs[i] < lows[i - 2]
        if bearish_gap and session[i]:
            gap_pct = (lows[i - 2] - highs[i]) / lows[i - 2] * 100.0
            if gap_pct >= fvg_threshold:
                fvg_array.insert(0, { ... })   # unchanged body
```
  Keep everything else identical. Update the docstring.

- [ ] **Step 3: Write `tests/test_ifvg_threshold.py` (failing).** On a synthetic in-session frame with a small gap and a large gap: `fvg_threshold=0.0` admits both (states change), a threshold above the small gap's pct but below the large gap's pct admits only the large one → assert the small-gap case yields `"None"` (no FVG created) while the large-gap case still inverts. Also assert monotonicity: number of non-"None" bars at a higher threshold ≤ that at threshold 0 on the same data.

- [ ] **Step 4: Run `pytest tests/test_ifvg_threshold.py -q`.** Expected: FAIL then (after Step 2) PASS.

- [ ] **Step 5: Split + parameterize `backtest/engine.py`.** Refactor so behavior is byte-identical for defaults but parameters are threaded:
  - `compute_signal_layer(df, params)`: `in_sess = in_session_mask(df.index, params.session_start, params.session_end)`; `ifvg = compute_ifvg(df, in_sess, params.fvg_threshold)`; `cisd = compute_cisd(df)`; `ema = compute_ema(df, params.ema_length)`; `sig = double_confirmation(ifvg, cisd)`. Return a dict of the numpy arrays (`sig, ema_v, sess, o, h, l, c`, `days = df.index.tz_convert("America/New_York").date`, `index = df.index`).
  - `run_execution(layer, params, fill_mode="next_open")`: the existing bar loop, but `SWING → params.swing_lookback`, `RR → params.rr`, and the loop starts at `range(params.swing_lookback, n)`. Keep `PT_VALUE=20.0`, `MAX_TRADES_PER_DAY=1`, stop-first/gap-through `_try_exit`, next-open fill, and the same-bar-span counter (now attached to `run_execution`).
  - `backtest(df, params=StrategyParams(), fill_mode="next_open")` = `run_execution(compute_signal_layer(df, params), params, fill_mode)`. Preserve `backtest.same_bar_span_count`.

- [ ] **Step 6: Write `tests/test_params.py`.** Assert `StrategyParams()` has the documented defaults; assert it is frozen (`hash` works / assignment raises).

- [ ] **Step 7: Write `tests/test_engine_default_regression.py` (the critical lock).** Load a FIXED slice of real data (deterministic) and assert `backtest(df_slice, StrategyParams())` equals `backtest(df_slice)` (Phase-2 call path) trade-for-trade (entry_time, direction, entry, stop, target, exit, pnl_usd). Prefer a slice (e.g. 3 months) for speed. Additionally, mark a slow full-data check (skippable) asserting the full run still yields **605 trades** with aggregate PF ≈ 0.85886 — OR document that the committed `backtest_results.json` (605 / PF 0.8589) is the regression anchor and assert the full run reproduces its `n_generated` and `profit_factor` when raw data is present (skip if absent).

- [ ] **Step 8: Run `pytest tests/ -q`.** Expected: all green (Phase-2 tests unchanged + new). **Commit:** `feat: parameterize engine via StrategyParams (defaults regression-locked)`.

---

## Task 2: Grid + walk-forward optimizer

**Files:** Create `tuning/__init__.py`, `tuning/grid.py`, `tuning/walkforward.py`; Test `tests/test_grid.py`, `tests/test_walkforward.py`.

**Interfaces:**
- Produces: `build_grid() -> list[StrategyParams]` (144 combos incl. defaults); `make_folds() -> list[Fold]` (`Fold` = train_start/train_end/test_start/test_end dates); `select_params(is_results) -> StrategyParams` (max-PF w/ trade floor + tie-breaks + default fallback); `walk_forward(df, grid, folds) -> dict` (per-fold selected params + IS/OOS metrics + stitched OOS trades + default-on-same-OOS metrics).

- [ ] **Step 1: Write `tuning/grid.py`.** `build_grid()` returns the Cartesian product of the four grids as `StrategyParams`; assert `StrategyParams()` (all defaults) is present. Constants `FVG_GRID, RR_GRID, EMA_GRID, SWING_GRID` at module top.

- [ ] **Step 2: Write `tests/test_grid.py`.** `len(build_grid()) == 144`; defaults are in the grid; no duplicate combos.

- [ ] **Step 3: Write `tests/test_walkforward.py` (failing) — the no-leakage + selection tests.**
  - `make_folds()`: 4 folds; every `test_start > train_end`; windows match the spec dates.
  - `select_params`: given synthetic in-sample results (list of `{params, profit_factor, n_trades, max_drawdown}`), picks the max-PF combo with `n_trades >= 20`; ignores a higher-PF combo that has `< 20` trades; applies tie-breaks; falls back to defaults when none meet the floor.
  - No-leakage: a `_window_trades(all_trades, start, end)` helper returns only trades whose entry_date is within `[start, end]`; assert a trade dated in the test window is never in the train slice.

- [ ] **Step 4: Run it.** Expected: FAIL.

- [ ] **Step 5: Implement `tuning/walkforward.py` (efficient precompute-and-slice).**
  - `make_folds()` returns the 4 `Fold`s from the Global Constraints dates.
  - **Precompute once over the FULL df:** `compute_cisd(df)` (param-free) once; `compute_ema(df, L)` for each `L in EMA_GRID`; `compute_ifvg(df, in_sess, t)` for each `(t in FVG_GRID)` with the default session mask; then `double_confirmation(ifvg_t, cisd)` for each needed combo. Cache in dicts keyed by the params that matter. (Session fixed for the base run, so one `in_sess`.)
  - For each fold and each `params` in the grid: build the signal layer by SLICING the precomputed arrays to the train window, run `run_execution` on the slice, compute IS metrics (`metrics.py`). `select_params` over the IS results → `best`. Then slice to the TEST window, run `run_execution` with `best` → OOS trades for that fold. Also run `run_execution` with `StrategyParams()` on the test slice → default-OOS baseline.
  - Return: per-fold `{fold, selected_params, is_pf, is_n, oos_metrics, oos_default_metrics, oos_trades}` + the stitched OOS trade list (all folds concatenated) + stitched OOS-default trades. Include `grid_size`, `min_is_trades`, and a `fallback_used` flag per fold.
  - Slicing helper: positional slice of `sig/ema_v/sess/o/h/l/c/days/index` to `[win_start_pos:win_end_pos]` by timestamp bounds; `run_execution` starts flat with `trades_today=0`.

- [ ] **Step 6: Run `pytest tests/test_grid.py tests/test_walkforward.py -q`.** Expected: PASS. **Commit:** `feat: coarse grid + walk-forward optimizer (no-leakage)`.

---

## Task 3: Runner + results.json + charts

**Files:** Create `run_phase4.py`; outputs `phase4_results.json` (committed), `charts/phase4_*.png` (committed). Test: `tests/test_smoke_phase4.py`.

- [ ] **Step 1: Write `run_phase4.py`.** `load_nq()` → `walk_forward(df, build_grid(), make_folds())`. Assemble `phase4_results.json`: `grid_size`, `min_is_trades`, `folds` (each: window dates, selected params, IS PF/n, OOS metrics, default-OOS metrics, fallback_used), **stitched OOS**: tuned aggregate (PF/WR/total_pnl/max_drawdown/n) vs default aggregate on the same OOS span, the **in-sample→OOS PF decay** per fold, and a **parameter-stability** summary (per-parameter set of selected values across folds). Time the run and record `run_seconds`.
- [ ] **Step 2: Charts (matplotlib → `charts/`):** (1) stitched OOS equity curve, tuned vs default; (2) per-fold OOS profit factor, tuned vs default (bars); (3) parameter-stability table/heatmap (selected value per param per fold); (4) in-sample vs OOS PF per fold (the overfitting-gap chart).
- [ ] **Step 3: Run `python run_phase4.py`.** Requires the raw data. Expected: writes JSON + PNGs, prints the headline (tuned-OOS PF vs default-OOS PF, and whether tuning helped). If runtime is long, that is acceptable — record it.
- [ ] **Step 4: `tests/test_smoke_phase4.py`** — run `walk_forward` on a SMALL grid (e.g. 4 combos) over a short real slice; assert it returns the expected structure and that per-fold test windows are strictly after train windows; skip if raw data absent.
- [ ] **Step 5: Run `pytest tests/ -q`.** Green. **Commit:** `feat: phase 4 runner + results.json + charts`.

---

## Task 4: Notebook + writeup + final review/merge

**Files:** Create `notebooks/04_parameter_tuning.ipynb`, `WRITEUP_PHASE4.md`; update `README.md`.

- [ ] **Step 1: `WRITEUP_PHASE4.md`** — lead with the honest OOS finding: **did walk-forward tuning beat the default out-of-sample, yes or no**, with the tuned-OOS vs default-OOS PF/WR/PnL. Show the overfitting gap (in-sample PF vs OOS PF) and the parameter stability (did the winners agree across folds?). State the multiple-testing disclosure (144 combos/fold). If the result is a null (no robust OOS improvement), say so plainly — it is a legitimate finding and consistent with the Monte-Carlo/ML studies' honesty. Restate the back-adjusted-data caveat once. Tie back to Phase 2: does tuning recover the real logs' selectivity?
- [ ] **Step 2: `notebooks/04_parameter_tuning.ipynb`** — self-contained: load data, run (or load) the walk-forward, show the fold table + 4 charts, narrate. Executes clean top-to-bottom.
- [ ] **Step 3: Update `README.md`** — add the Phase-4 section (what it does, `python run_phase4.py`, the headline OOS verdict).
- [ ] **Step 4: Final whole-branch review** (superpowers:requesting-code-review over `git diff <merge-base>..HEAD`); fix Critical/Important.
- [ ] **Step 5:** `pytest tests/ -q` green → **merge `feat/phase4-parameter-tuning` → master** (do NOT push to GitHub — held for the user) → update the vault (`15-fyp-strategy-engine/_INDEX.md` Phase-4 done + the honest OOS result; program now complete).

---

## Self-review notes
- **Spec coverage:** parameterization+regression (T1), grid+walk-forward+no-leakage+selection (T2), OOS-only reporting+overfitting/stability diagnostics (T3), honest writeup+merge (T4) — all mapped.
- **Anti-overfitting integrity:** headline is stitched OOS; params selected on in-sample only; no-leakage tested; grid size disclosed; parameter stability + in-sample→OOS decay reported; tuning objective is the strategy's own OOS PF, never agreement with the real logs.
- **Correctness locks:** default `StrategyParams` reproduces Phase-2 trade-for-trade (regression test); `fvg_threshold=0` is behavior-preserving; precompute-and-slice uses only causal past data (valid, no future leakage).
- **Scrutinize in final review:** the slice-by-timestamp alignment in walk-forward (positional vs index), the no-leakage boundary (train_end < test_start, inclusive/exclusive), and the default-fallback path when a fold has < 20 in-sample trades.
