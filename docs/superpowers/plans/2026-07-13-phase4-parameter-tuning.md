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
- **Objective (in-sample selection):** max **profit factor**, subject to **≥ `MIN_IS_TRADES = 50`** in-sample trades; tie-break higher trade count, then lower max drawdown. If no combo meets the floor in a fold, select the DEFAULT params for that fold (documented fallback) and flag it. (Empirically the default fires ~200 IS trades/window, so the fallback is not expected to trigger under the base grid — but it is handled and reported.)
- **Headline = stitched OOS only.** In-sample numbers appear solely to show the in-sample→OOS decay. Report grid size, per-fold selected params, parameter stability, and OOS-tuned vs OOS-default.
- **Data caveat (from Phase 2, restate once):** `load_nq` is back-adjusted; all comparisons here are strategy-vs-strategy on the same series, so signals/outcomes/relative-PnL are valid. Real logs are a sanity reference only, never the tuning target.
- **Reuse:** `metrics.py` (`profit_factor, win_rate, total_pnl, max_drawdown`); Phase-2 `strategy/` + `backtest/`.
- **PRE-REGISTRATION FREEZE (anti-meta-overfitting).** `FVG_GRID, RR_GRID, EMA_GRID, SWING_GRID`, the four fold date-windows, `MIN_IS_TRADES`, and the selection objective are committed as constants in `tuning/grid.py` + `tuning/walkforward.py` **before** `run_phase4.py` is executed. The runner is **single-shot**: the stitched OOS number is observed exactly once. `run_phase4.py` records a **SHA-256 hash of the frozen config + the current git commit SHA** into `phase4_results.json`. Any change to grid/folds/objective/floor **after** seeing an OOS result requires a NEW dated spec and a re-run explicitly labelled a new experiment — never a silent overwrite. Tuning is optimized against the strategy's own **in-sample** PF only, never against the OOS headline or the real logs.
- **PRE-REGISTERED SUCCESS RULE (falsifiable + null control).** A **positive** ("tuning robustly helps") verdict requires ALL of: (a) tuned stitched-OOS PF **> 1.0** (absolute profitability floor); (b) tuned OOS PF exceeds default OOS PF by **≥ 0.10**; (c) tuned beats default OOS PF in **≥ 3 of 4 folds**; (d) tuned OOS PF exceeds the **median OOS PF of all 144 combos** (the selection-luck null — if a random combo does about as well OOS, "selection" was luck). Anything less is reported as **null / inconclusive**. This rule is fixed BEFORE the run.
- **Null control (record every combo's OOS too).** For each fold and stitched, `run_phase4.py` records the OOS PF of **all 144 combos** (not just the selected one), so the report can show the selected pick's **OOS percentile rank** vs the full distribution and the **median-combo** null. Also record one **randomly-drawn combo's** OOS as a sanity anchor.
- **Fold boundaries are HALF-OPEN:** train = `[train_start 00:00, test_start 00:00)`, test = `[test_start 00:00, next_boundary 00:00)`. Slicing uses `index.searchsorted(ts, side="left")` on the tz-aware index; assert `n_slice == b - a` and `train_slice.index.max() < test_slice.index.min()` per fold, and that the 4 test windows are contiguous, non-overlapping, and cover 2024-01..2025-12-11.
- **Selection floor `MIN_IS_TRADES = 50`** (raised from 20). PF on ~20 trades under a 144-way max is noise-dominated; the default already fires ~200 in-sample trades per 12-month window, so a 50-trade floor still admits real combos while rejecting lucky low-N picks. Report a **selection-robustness diagnostic**: how many combos' in-sample PF fall within a bootstrap CI of the winner's (many ⇒ selection is near coin-flip; disclose).
- **fvg_threshold caveat (back-adjustment confound):** `fvg_threshold` is a **percent** gap (`gap/price*100`); on the back-adjusted series the price *level* drifts (~+2655 pts in early 2023 → ~0 by 2025), so a fixed % maps to a **different point-gap filter across time**. Disclose this in the writeup and interpret `fvg_threshold` results cautiously (its effective strictness is not constant train→test).
- **n = 4 folds is DESCRIPTIVE, not inferential** — 4 paired OOS observations, no statistical power; and consecutive train windows overlap 6 months, so the 4 selected param-sets are **not independent**, which *inflates* apparent parameter stability. State both explicitly; do not claim significance.
- **Boundary artifacts (state + handle):** a trade still open at a window's `test_end` is **dropped** from the stitched OOS (the next fold starts flat); no trade spans a window boundary or a contract roll (1 trade/day, intraday), so per-trade $ and PF **are** back-adjustment-safe for the OOS comparison (only the fvg% base above is confounded). Precompute-then-slice is **causal/leak-free** (compute_cisd/ifvg/ema/double_confirmation each read only bars ≤ i — verified in review).

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

- [ ] **Step 7: Golden-trade regression lock (NOT a self-comparison).** A `backtest(df_slice, StrategyParams()) == backtest(df_slice)` test is tautological — same code path, always passes even if the refactor changed every trade. Instead:
  - **Step 7a (do this BEFORE any Task-1 code change — capture the ORIGINAL engine's output):** on a FIXED deterministic real slice (e.g. `load_nq().loc["2024-05-01":"2024-07-31"]`), run the **current committed (pre-refactor) `backtest`** and serialize every trade's `entry_time, direction, entry, stop, target, exit, pnl_usd, r_multiple` to `tests/fixtures/phase2_golden_trades.json` (committed). This fixture is the immutable Phase-2 anchor.
  - **Step 7b:** `tests/test_engine_default_regression.py` asserts `backtest(df_slice, StrategyParams())` reproduces `phase2_golden_trades.json` **trade-for-trade, UNCONDITIONALLY** (no data-gate/skip — the fixture is committed, the slice is deterministic). If the raw data file is genuinely absent the test may skip, but the fixture itself is committed so CI on this machine runs it.
  - **Step 7c (secondary confirmation, skippable if raw data absent):** assert the FULL-data default run still yields **605 trades** and `profit_factor ≈ 0.85886` (matches committed `backtest_results.json`). This is a secondary check, not the primary lock.

- [ ] **Step 8: Run `pytest tests/ -q`.** Expected: all green (Phase-2 tests unchanged + new). **Commit:** `feat: parameterize engine via StrategyParams (defaults regression-locked)`.

---

## Task 2: Grid + walk-forward optimizer

**Files:** Create `tuning/__init__.py`, `tuning/grid.py`, `tuning/walkforward.py`; Test `tests/test_grid.py`, `tests/test_walkforward.py`.

**Interfaces:**
- Produces: `build_grid() -> list[StrategyParams]` (144 combos incl. defaults); `make_folds() -> list[Fold]` (`Fold` = train_start/train_end/test_start/test_end dates); `select_params(is_results) -> StrategyParams` (max-PF w/ trade floor + tie-breaks + default fallback); `walk_forward(df, grid, folds) -> dict` (per-fold selected params + IS/OOS metrics + stitched OOS trades + default-on-same-OOS metrics).

- [ ] **Step 1: Write `tuning/grid.py`.** `build_grid()` returns the Cartesian product of the four grids as `StrategyParams`; assert `StrategyParams()` (all defaults) is present. Constants `FVG_GRID, RR_GRID, EMA_GRID, SWING_GRID` at module top.

- [ ] **Step 2: Write `tests/test_grid.py`.** `len(build_grid()) == 144`; defaults are in the grid; no duplicate combos.

- [ ] **Step 3: Write `tests/test_walkforward.py` (failing) — the no-leakage + selection tests.**
  - `make_folds()`: 4 folds; **half-open** windows matching the spec dates; assert per fold `train_end == test_start` boundary is handled as `[train_start, test_start)` / `[test_start, next)` (a Dec-31 09:30 session bar belongs to that year's train, not lost); assert the 4 test windows are **contiguous, non-overlapping**, and cover 2024-01..2025-12-11.
  - **Positional-slice alignment (the 1-bar-leak guard):** for a fold, slice the precomputed full-series arrays to the train and test windows via `index.searchsorted`; assert `sliced.index equals df.loc[window].index` exactly, `np.array_equal(sliced['c'], df.loc[window]['close'].to_numpy())`, `n_slice == b-a`, and `train_slice.index.max() < test_slice.index.min()`. A +1 error here would carry the next bar's signal/EMA into every decision — this test makes that impossible.
  - `select_params`: given synthetic in-sample results (`{params, profit_factor, n_trades, max_drawdown}`), picks the max-PF combo with `n_trades >= 50`; **ignores** a higher-PF combo with `< 50` trades; applies tie-breaks (trade count, then lower max drawdown); falls back to defaults when none meet the floor (and sets `fallback_used`).
  - No-leakage on trades: a `_window_trades(all_trades, start, end)` helper returns only trades whose entry_date is in the half-open window; assert a test-window trade is never in the train slice.

- [ ] **Step 4: Run it.** Expected: FAIL.

- [ ] **Step 5: Implement `tuning/walkforward.py` (efficient precompute-and-slice).**
  - `make_folds()` returns the 4 `Fold`s from the Global Constraints dates.
  - **Precompute once over the FULL df:** `compute_cisd(df)` (param-free) once; `compute_ema(df, L)` for each `L in EMA_GRID`; `compute_ifvg(df, in_sess, t)` for each `(t in FVG_GRID)` with the default session mask; then `double_confirmation(ifvg_t, cisd)` for each needed combo. Cache in dicts keyed by the params that matter. (Session fixed for the base run, so one `in_sess`.)
  - For each fold and each `params` in the grid: build the signal layer by SLICING the precomputed arrays to the train window, run `run_execution` on the slice, compute IS metrics (`metrics.py`). `select_params` over the IS results → `best`. Then slice to the TEST window and run `run_execution` with `best` → OOS trades for that fold; also with `StrategyParams()` → default-OOS baseline.
  - **Null control (required):** run **all 144 combos** on the TEST slice too and record each combo's OOS PF → the fold's `oos_pf_distribution` (list of 144). Compute the selected pick's **OOS percentile rank** and the **median-combo OOS PF**. Record one **fixed-seed random combo's** OOS PF as a sanity anchor (derive the index deterministically from the fold number, e.g. `(fold_i * 37) % 144`, since `Math.random`/time are unavailable and must not be used).
  - **Selection-robustness diagnostic:** on the IS results, bootstrap a CI around the winner's IS PF (resample its IS trades) and count how many other combos' IS PF fall inside it → `n_combos_within_winner_ci` (large ⇒ selection is near coin-flip; disclosed).
  - Return: per-fold `{fold, train/test dates, selected_params, is_pf, is_n, oos_metrics, oos_default_metrics, oos_trades, oos_pf_distribution, selected_oos_percentile, median_combo_oos_pf, random_combo_oos_pf, n_combos_within_winner_ci, fallback_used}` + stitched OOS trades (tuned) + stitched OOS trades (default). Include `grid_size`, `min_is_trades`, and the frozen-config hash + git SHA.
  - Slicing helper: half-open positional slice of `sig/ema_v/sess/o/h/l/c/days/index` via `index.searchsorted(ts, side="left")`; `run_execution` starts flat with `trades_today=0`. A trade open at `test_end` is dropped (next fold starts flat).

- [ ] **Step 6: Run `pytest tests/test_grid.py tests/test_walkforward.py -q`.** Expected: PASS. **Commit:** `feat: coarse grid + walk-forward optimizer (no-leakage)`.

---

## Task 3: Runner + results.json + charts

**Files:** Create `run_phase4.py`; outputs `phase4_results.json` (committed), `charts/phase4_*.png` (committed). Test: `tests/test_smoke_phase4.py`.

- [ ] **Step 1: Write `run_phase4.py` (single-shot).** `load_nq()` → `walk_forward(df, build_grid(), make_folds())`. Compute a **SHA-256 of the frozen config** (`build_grid()` combos + fold dates + `MIN_IS_TRADES` + objective name) and read the current **git commit SHA**; write both into the JSON. Assemble `phase4_results.json`:
  - `config_hash`, `git_sha`, `grid_size` (144), `min_is_trades` (50), `run_seconds`.
  - `folds`: each with window dates, selected params, IS PF/n, OOS metrics, default-OOS metrics, `selected_oos_percentile`, `median_combo_oos_pf`, `random_combo_oos_pf`, `n_combos_within_winner_ci`, `fallback_used`.
  - **stitched OOS:** tuned aggregate (PF/WR/total_pnl/max_drawdown/n) vs default aggregate on the same OOS span — reported **both over all 4 folds AND excluding any fallback folds** (with the fallback-fold count stated).
  - **`success_rule`:** evaluate the 4 pre-registered conditions (tuned OOS PF > 1.0; tuned − default ≥ 0.10; tuned beats default in ≥ 3/4 folds; tuned OOS PF > median-combo OOS PF) and a boolean `robust_improvement` = all four true, plus each sub-condition's value. This is the falsifiable verdict.
  - `in_sample_to_oos_decay` per fold (IS PF vs the selected combo's OOS PF); `parameter_stability` (per-parameter set of selected values across folds) **with the caveat that overlapping train windows inflate agreement** recorded alongside.
- [ ] **Step 2: Charts (matplotlib → `charts/`):** (1) stitched OOS equity curve, tuned vs default; (2) per-fold OOS profit factor, tuned vs default (bars); (3) parameter-stability table/heatmap (selected value per param per fold); (4) in-sample vs OOS PF per fold (the overfitting-gap chart); (5) **the selection-luck null** — per fold (or stitched), a histogram of all 144 combos' OOS PF with the selected pick and the median-combo null marked, so the reader sees whether the tuned pick is special or just one of the pack.
- [ ] **Step 3: Run `python run_phase4.py`.** Requires the raw data. Expected: writes JSON + PNGs, prints the headline (tuned-OOS PF vs default-OOS PF, and whether tuning helped). If runtime is long, that is acceptable — record it.
- [ ] **Step 4: `tests/test_smoke_phase4.py`** — run `walk_forward` on a SMALL grid (e.g. 4 combos) over a short real slice; assert it returns the expected structure and that per-fold test windows are strictly after train windows; skip if raw data absent.
- [ ] **Step 5: Run `pytest tests/ -q`.** Green. **Commit:** `feat: phase 4 runner + results.json + charts`.

---

## Task 4: Notebook + writeup + final review/merge

**Files:** Create `notebooks/04_parameter_tuning.ipynb`, `WRITEUP_PHASE4.md`; update `README.md`.

- [ ] **Step 1: `WRITEUP_PHASE4.md`** — lead with the **pre-registered falsifiable verdict**: report `success_rule.robust_improvement` (yes/no) and each of the 4 sub-conditions with numbers (tuned OOS PF vs 1.0; tuned − default margin; fold win count; tuned vs the **median-combo null**). Show the overfitting gap (IS PF vs OOS PF per fold), the selected pick's **OOS percentile** among all 144 combos (is it special or lucky?), and the parameter stability **with the caveat that overlapping train windows inflate agreement and n=4 has no statistical power**. State the multiple-testing disclosure (144 combos/fold), the pre-registration freeze (config hash + git SHA), and the `fvg_threshold` back-adjustment caveat. If it's a **null** (very possible — the default itself is already a marginal ~0.9 OOS PF), say so plainly; a rigorously-measured null is a legitimate, portfolio-grade result exactly like the Monte-Carlo and ML studies. Tie back to Phase 2: does the tuned strategy move toward the real logs' selectivity (fewer, better trades)?
- [ ] **Step 2: `notebooks/04_parameter_tuning.ipynb`** — self-contained: load data, run (or load) the walk-forward, show the fold table + 4 charts, narrate. Executes clean top-to-bottom.
- [ ] **Step 3: Update `README.md`** — add the Phase-4 section (what it does, `python run_phase4.py`, the headline OOS verdict).
- [ ] **Step 4: Final whole-branch review** (superpowers:requesting-code-review over `git diff <merge-base>..HEAD`); fix Critical/Important.
- [ ] **Step 5:** `pytest tests/ -q` green → **merge `feat/phase4-parameter-tuning` → master** (do NOT push to GitHub — held for the user) → update the vault (`15-fyp-strategy-engine/_INDEX.md` Phase-4 done + the honest OOS result; program now complete).

---

## Self-review notes
- **Spec coverage:** parameterization+golden-regression (T1), grid+walk-forward+no-leakage+selection+null (T2), OOS-only reporting+success-rule+overfitting/stability diagnostics (T3), honest falsifiable writeup+merge (T4) — all mapped.
- **This plan was revised after a 3-lens adversarial review (leakage rigor / parameterization / methodology) that returned GO-WITH-FIXES. The 3 blockers are now closed in-plan:** (1) **pre-registration freeze** — grid/folds/objective/floor frozen as constants + config-hash + git-SHA in the JSON, single-shot run; (2) **falsifiable success rule + selection-luck null** — a compound PF>1 / margin / ≥3-of-4-folds / beats-median-combo rule, with all 144 combos' OOS PF recorded; (3) **golden-trade regression** — a fixture captured from the ORIGINAL engine before refactor, asserted unconditionally (not a self-comparison).
- **Important items folded in:** half-open fold boundaries + disjointness/contiguity tests; positional-slice alignment test (1-bar-leak guard); `MIN_IS_TRADES` raised to 50 + selection-robustness diagnostic; `fvg_threshold` back-adjustment caveat; n=4 stated descriptive-not-inferential with overlapping-window stability caveat.
- **Anti-overfitting integrity:** headline is the pre-registered OOS verdict; params selected on in-sample only; no-leakage is structural (tested); tuning never optimized against the OOS number or the real logs.
- **Scrutinize in final review:** the golden fixture truly came from the pre-refactor engine; the searchsorted slice alignment; the success-rule arithmetic; and that `run_phase4.py` is genuinely single-shot (config hash present).
