# Phase 5 — Costs + Exits + Volatility Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Test, pre-registered and walk-forward, whether realistic costs + smarter exits + a volatility filter make the strategy net-profitable out-of-sample — or return an honest null.

**Architecture:** Extend the Phase-2/4 engine with a pluggable exit handler (5 modes), an ATR volatility gate, and a cost model — all off/default by default so Phase-2/4 behavior is byte-preserved. Sweep only `exit_mode × vol_filter` (20 combos, entry params fixed at defaults) through the reused Phase-4 walk-forward, selecting on **net** in-sample PF. Report the net success rule + cost sensitivity + selection-luck null.

**Tech Stack:** Python, pandas 2.2.3, numpy 2.1.3, pytest 8.3.3, matplotlib. Builds on `strategy/`, `backtest/`, `tuning/`.

## Global Constraints

- **Behavior preservation:** with `exit_mode="fixed_1_5R"`, `vol_filter="off"`, and costs OFF, `run_execution` MUST reproduce the Phase-2 golden fixture (`tests/fixtures/phase2_golden_trades.json`) trade-for-trade. This is the primary regression lock (same pattern as Phase 4 Task 1).
- **`StrategyParams` extension (defaults behavior-preserving):** add `exit_mode: str = "fixed_1_5R"`, `vol_filter: str = "off"`. Costs are a SEPARATE `CostModel` (not in StrategyParams), OFF by default, so gross Phase-2/4 paths are untouched.
- **Cost model (pre-registered):** `TICK_VALUE=5.0`, `COMMISSION_RT=5.0`, `SLIPPAGE_TICKS_ENTRY=1`, `SLIPPAGE_TICKS_STOP=1`. `net = gross - COMMISSION_RT - TICK_VALUE*(SLIPPAGE_TICKS_ENTRY + (SLIPPAGE_TICKS_STOP if exit_reason=="stop" else 0))`. Target/limit exits pay no stop-slippage. Sensitivity multipliers `{0.0, 1.0, 2.0}` scale commission+slippage (reported, never used for selection).
- **Exit modes (5, pre-registered):** initial stop = swing_lookback-bar swing, R = |entry−stop|.
  1. `fixed_1_5R`: target = entry ± 1.5R. (base)
  2. `breakeven_1R`: once price reaches +1R intrabar, stop → entry; target 1.5R.
  3. `trail_swing`: once +1R reached, stop trails the rolling swing (swing_lookback bars); NO fixed target — exit only on the trailing stop (initial stop applies before +1R).
  4. `partial_1R`: at +1R, realize 0.5 unit at +1R, move remainder (0.5 unit) stop → breakeven, remainder target = 3R. Trade `pnl = 0.5*R_banked_usd + 0.5*remainder_pnl_usd` (fractional units allowed in sim).
  5. `time_stop`: `fixed_1_5R` plus a hard exit at market on the first bar with ET time ≥ `TIME_STOP_ET="11:00"` if still open.
  - Same-bar sequencing: keep the existing **stop-first / gap-through** assumption; when a mode needs "+1R reached", test it against the bar high/low BEFORE applying stop/target for that bar, in a fixed documented order. `exit_reason ∈ {"stop","target","trail","time","partial+..."}` recorded for costing.
- **Volatility filter (leak-free, scale-free):** `ATR14` = ATR over trailing 14 one-minute bars at the signal bar. `vol_filter ∈ {off, p25, p50, p75}`; the threshold is the p-th percentile of signal-bar ATR14 over the **train window only** (per fold). `off` = no gate. Enter only if `ATR14 ≥ threshold`.
- **Grid = 20:** `EXIT_MODES` (5) × `VOL_FILTERS` (4); entry params fixed at `StrategyParams()` defaults. `(fixed_1_5R, off)` = the base baseline; it MUST be in the grid.
- **Walk-forward:** reuse Phase-4 folds/precompute-slice/no-leakage machinery. Selection: max **net** in-sample PF, `MIN_IS_TRADES=50`, tie-break higher net total_pnl then lower max_drawdown. Null: all 20 combos' net OOS PF per fold + percentile + median-combo null.
- **Pre-registered net success rule:** positive iff ALL of — (a) tuned net stitched-OOS PF > 1.0; (b) tuned − base net OOS PF ≥ 0.10; (c) tuned beats base net in ≥ 3/4 folds; (d) tuned net OOS PF > median of the 20 combos' net OOS PF. Else null. Frozen; config hashed + git SHA; single-shot.
- **Reuse:** `metrics.py`; `tuning/walkforward.py` helpers (folds, `_slice_layer`, `select_params` generalized); the engine.

---

## Task 1: ATR + costs + exit modes (regression-locked)

**Files:** Create `strategy/atr.py`, `backtest/costs.py`, `backtest/exits.py`; Modify `strategy/params.py`, `backtest/engine.py`. Test: `tests/test_atr.py`, `tests/test_costs.py`, `tests/test_exits.py`, `tests/test_engine_p5_regression.py`.

**Interfaces:**
- Produces: `compute_atr(df, period=14) -> pd.Series` (causal); `CostModel(commission_rt, slippage_ticks_entry, slippage_ticks_stop, tick_value, multiplier=1.0)` + `apply_costs(gross, exit_reason) -> net`; the 5 exit-mode handlers in `exits.py`; `StrategyParams` with `exit_mode`, `vol_filter`; `run_execution(layer, params, fill_mode="next_open", cost_model=None, atr=None, vol_threshold=None) -> list[Trade]` (Trade gains `net_pnl`, `exit_reason`).

- [ ] **Step 1: `strategy/atr.py`.** `compute_atr(df, period=14)`: `TR = max(high-low, |high-prev_close|, |low-prev_close|)`; ATR = Wilder's RMA (or SMA of TR — pick RMA, document). Causal (uses bars ≤ i). Returns a Series aligned to df.index. Test `tests/test_atr.py`: TR/ATR values on a tiny hand frame; first value NaN handling; no look-ahead (a future bar can't change an earlier ATR).

- [ ] **Step 2: `backtest/costs.py`.** `CostModel` dataclass (frozen) with the pre-registered defaults + `multiplier`; `net_pnl(gross, exit_reason)` applies commission + entry slippage always, + stop slippage only when `exit_reason=="stop"`, all × multiplier. Test `tests/test_costs.py`: a target-exit winner pays `COMMISSION_RT + 1*TICK_VALUE`; a stop-exit loser pays `COMMISSION_RT + 2*TICK_VALUE`; multiplier 0 → net==gross; multiplier 2 → double the cost.

- [ ] **Step 3: `backtest/exits.py` (failing tests first).** Write `tests/test_exits.py` with a hand-built per-bar sequence for EACH mode asserting the exact exit price/reason:
  - `fixed_1_5R`: hits target at 1.5R → `("target", target_price)`; hits stop → `("stop", stop_price)`.
  - `breakeven_1R`: after a bar reaches +1R, a later bar returning to entry exits at entry as `("stop", entry)` with ~0 gross (before costs); if it never reaches +1R the original stop applies.
  - `trail_swing`: after +1R, price pulling back to the trailing swing exits `("trail", swing_level)`; a runner beyond 1.5R keeps running (no fixed-target cap).
  - `partial_1R`: reaching +1R banks 0.5 unit at +1R; remainder to 3R or breakeven; assert blended pnl for both remainder outcomes.
  - `time_stop`: still open at 11:00 ET → `("time", close_at_11:00)`.
  Then implement `exits.py` as a per-mode handler with explicit trade state (`activated_1R`, `stop_level`, `half_closed`, `remainder_stop`, `remainder_target`). Keep a single documented intrabar order (e.g. check +1R activation on the bar first, then stop-first exit).

- [ ] **Step 4: Extend `strategy/params.py`** — add `exit_mode: str = "fixed_1_5R"`, `vol_filter: str = "off"`. Test the new defaults + still frozen.

- [ ] **Step 5: Thread through `run_execution`.** Add optional `cost_model`, `atr`, `vol_threshold`. On a signal: if `vol_threshold is not None` and `atr[i] < vol_threshold`, skip the entry. On managing an open trade, delegate to the exit handler for `params.exit_mode`. On close, set `trade.exit_reason` and, if `cost_model`, `trade.net_pnl = cost_model.net_pnl(trade.pnl_usd, trade.exit_reason)` (else `net_pnl = pnl_usd`). The signal-layer + entry logic is otherwise unchanged.

- [ ] **Step 6: `tests/test_engine_p5_regression.py` (primary lock).** `run_execution` with `exit_mode="fixed_1_5R"`, `vol_filter` off (no atr/threshold), `cost_model=None` reproduces `tests/fixtures/phase2_golden_trades.json` trade-for-trade (entry_time/direction/entry/stop/target/exit/pnl_usd). UNCONDITIONAL (fixture committed). This proves the exit refactor didn't change the base path.

- [ ] **Step 7: Run `pytest tests/ -q`.** All green (every Phase-2/4 test still passes). **Commit:** `feat: ATR + cost model + 5 exit modes (base path regression-locked)`.

---

## Task 2: Grid + net walk-forward (vol percentiles)

**Files:** Create `tuning/grid_p5.py`, `tuning/walkforward_p5.py`; Test `tests/test_grid_p5.py`, `tests/test_walkforward_p5.py`.

**Interfaces:** `build_grid_p5() -> list[StrategyParams]` (20 combos, entry fixed at defaults); `walk_forward_p5(df, grid, folds, cost_model) -> dict` (per-fold net metrics + selected + all-20 net-OOS null + base-net baseline).

- [ ] **Step 1: `tuning/grid_p5.py`.** `EXIT_MODES=("fixed_1_5R","breakeven_1R","trail_swing","partial_1R","time_stop")`, `VOL_FILTERS=("off","p25","p50","p75")`; `build_grid_p5()` = 20 `StrategyParams(exit_mode=…, vol_filter=…)` (all other fields default). Assert `(fixed_1_5R, off)` present. Test `tests/test_grid_p5.py` (20, base present, no dups).

- [ ] **Step 2: `tests/test_walkforward_p5.py` (failing).** Vol-filter leak-free test: per fold, the ATR percentile threshold for `p50` equals `np.percentile(train_window_signal_ATR, 50)` and NO test-window ATR is used; a synthetic assert that changing only test-window ATR leaves the threshold unchanged. Selection uses NET pf. Reuse Phase-4 no-leakage fold asserts.

- [ ] **Step 3: Implement `tuning/walkforward_p5.py`.** Reuse `tuning/walkforward.py`'s `make_folds`, `_precompute` (add ATR to the precompute), `_slice_layer`, and the selection/null structure — thin-wrap or import. Per fold: compute the signal-bar ATR series once; for each `vol_filter`, the threshold = percentile of TRAIN-window signal-bar ATR (p-th; `off` → None). Run each of the 20 combos with the `cost_model` → NET metrics. `select_params` on NET in-sample PF (`MIN_IS_TRADES=50`, tie-breaks). OOS: selected + all-20 null + base `(fixed_1_5R, off)` net baseline. Cap net PF (inf) like Phase 4. Return the per-fold + stitched net tuned/base trade lists + grid_size.

- [ ] **Step 4: Run `pytest tests/test_grid_p5.py tests/test_walkforward_p5.py -q`.** Green. **Commit:** `feat: phase 5 grid + net walk-forward (leak-free ATR percentiles)`.

---

## Task 3: Runner + results.json + charts

**Files:** Create `run_phase5.py`; outputs `phase5_results.json` + `charts/phase5_*.png`. Test `tests/test_smoke_phase5.py`.

- [ ] **Step 1: `run_phase5.py` (single-shot).** `load_nq()` → `walk_forward_p5(df, build_grid_p5(), make_folds(), CostModel())`. `config_hash` (SHA-256 of grid + folds + MIN_IS_TRADES + cost constants + objective) + `git_sha`. Assemble `phase5_results.json`: config_hash, git_sha, grid_size(20), cost model, per-fold (selected exit_mode+vol_filter, net IS PF/n, net OOS metrics, base-net metrics, percentile, median-combo net null, fallback_used), stitched net OOS tuned vs base (all folds + excl. fallback), the **success_rule** (4 conditions + `robust_improvement`), and a **cost-sensitivity** block re-computing the stitched tuned & base net PF/total_pnl at multipliers 0×/1×/2×. run_seconds.
- [ ] **Step 2: Charts (`charts/phase5_*.png`):** (1) stitched net OOS equity: tuned vs base vs gross; (2) per-fold net OOS PF tuned vs base; (3) selected exit_mode + vol_filter per fold (stability); (4) cost-sensitivity bars (net total_pnl at 0×/1×/2×); (5) 20-combo net-OOS-PF null with the selected pick + median marked.
- [ ] **Step 3: Run `python run_phase5.py` SYNCHRONOUSLY (foreground, wait ~5-15 min).** Commit `phase5_results.json` + PNGs.
- [ ] **Step 4: `tests/test_smoke_phase5.py`** — `walk_forward_p5` on a small grid + short real slice; structure + net metrics present; skip if raw data absent.
- [ ] **Step 5: `pytest tests/ -q` green.** **Commit:** `feat: phase 5 runner + results.json + charts`.

---

## Task 4: Notebook + writeup + final review/merge

- [ ] **Step 1: `WRITEUP_PHASE5.md`** — lead with the pre-registered NET verdict (`robust_improvement` yes/no) + the 4 conditions with numbers; then the sub-story (which exit mode / vol filter got selected, did costs flip a gross-positive to net-negative, did the loss narrow like Phase 4); the cost-sensitivity (does the verdict survive 0×→2× costs?); the selection-luck null; the overfitting caveats (20 combos, n=4 folds descriptive, overlapping windows); pre-registration disclosure. If null, state it plainly. Update the "program" framing: Phase 5 is the last credible lever tried.
- [ ] **Step 2: `notebooks/05_costs_exits_volfilter.ipynb`** — load `phase5_results.json` (don't re-run the sweep), show fold table + 5 charts + the cost-sensitivity, narrate. Executes clean.
- [ ] **Step 3: Update `README.md`** — Phase 5 section + headline net verdict.
- [ ] **Step 4: Final whole-branch review** (superpowers:requesting-code-review over the branch diff); fix Critical/Important.
- [ ] **Step 5:** `pytest tests/ -q` green → merge `feat/phase5-exits-costs-volfilter` → master → vault update (`15-fyp-strategy-engine/_INDEX.md` Phase 5 + honest net result). **Do NOT push — held for the user.**

---

## Self-review notes
- **Spec coverage:** ATR+costs+exits regression-locked (T1), 20-combo net walk-forward + leak-free ATR percentiles (T2), single-shot runner + net success rule + cost sensitivity + null (T3), honest writeup+merge (T4).
- **Anti-overfitting:** entry params fixed (no re-search); only 20 combos; net-of-cost headline (higher bar); pre-registration + config hash; selection-luck null; walk-forward leak-free (reused, tested).
- **Correctness locks:** base path (fixed_1_5R/off/no-costs) reproduces Phase-2 golden fixture; ATR + vol percentile are leak-free (train-only); cost math unit-tested per exit_reason.
- **Scrutinize in final review:** the exit-mode intrabar sequencing (partial/trail/breakeven same-bar order), the ATR-percentile-from-train-only leak guard, the net-PF selection + inf cap, and the fractional-unit partial P&L math.
