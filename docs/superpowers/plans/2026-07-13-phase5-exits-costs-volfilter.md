# Phase 5 — Costs + Exits + Volatility Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Test, pre-registered and walk-forward, whether realistic costs + smarter exits + a volatility filter make the strategy net-profitable out-of-sample — or return an honest null.

**Architecture:** Extend the Phase-2/4 engine with a pluggable exit handler (5 modes), an ATR volatility gate, and a cost model — all off/default by default so Phase-2/4 behavior is byte-preserved. Sweep only `exit_mode × vol_filter` (20 combos, entry params fixed at defaults) through the reused Phase-4 walk-forward, selecting on **net** in-sample PF. Report the net success rule + cost sensitivity + selection-luck null.

**Tech Stack:** Python, pandas 2.2.3, numpy 2.1.3, pytest 8.3.3, matplotlib. Builds on `strategy/`, `backtest/`, `tuning/`.

## Global Constraints

- **Behavior preservation:** with `exit_mode="fixed_1_5R"`, `vol_filter="off"`, and costs OFF, `run_execution` MUST reproduce the Phase-2 golden fixture (`tests/fixtures/phase2_golden_trades.json`) trade-for-trade. This is the primary regression lock (same pattern as Phase 4 Task 1).
- **`StrategyParams` extension (defaults behavior-preserving):** add `exit_mode: str = "fixed_1_5R"`, `vol_filter: str = "off"`. Costs are a SEPARATE `CostModel` (not in StrategyParams), OFF by default, so gross Phase-2/4 paths are untouched.
- **Cost model (pre-registered) — reason-aware, no differential under-costing (Blocker 1):** `TICK_VALUE=5.0`, `COMMISSION_RT=5.0`, `SLIPPAGE_TICKS_ENTRY=1`, `SLIPPAGE_TICKS_EXIT=1`. Slippage is charged on **every market-order fill**, not only the literal `"stop"` reason. `MARKET_EXIT_REASONS = {"stop","trail","time"}` (adverse-side market fills) each pay 1 exit tick; **true limit fills** (`"target"`, the partial's +1R scale-out, the 3R remainder) pay 0 exit slippage. Every trade pays `COMMISSION_RT` + `SLIPPAGE_TICKS_ENTRY` tick. `partial_1R` has **two fills** → **two** commissions and slippage per leg by its own leg reason (the +1R scale-out is a limit = 0 exit tick; the remainder pays per its exit reason). Net is computed **per fill** and summed. Sensitivity multipliers `{0.0, 1.0, 2.0}` scale all commission+slippage (reported, never used for selection). *Note (disclosed): 1-tick stop slippage is optimistic for NQ fast-market stops; the 2× band is the realistic-pessimistic case and the writeup says so.*
- **Exit modes (5, pre-registered):** initial stop = swing_lookback-bar swing; **R and all managed levels (1R, 1.5R, 3R, breakeven) are anchored at the SIGNAL-BAR CLOSE**, i.e. `R = |signal_close − stop|` — exactly as the base engine computes risk/target (NOT the fill price). Coding R off `entry` would break the base-path regression.
  1. `fixed_1_5R`: target = entry ± 1.5R. (base)
  2. `breakeven_1R`: once price reaches +1R intrabar, stop → entry; target 1.5R.
  3. `trail_swing`: once +1R reached, stop trails the rolling swing (swing_lookback bars); NO fixed target — exit only on the trailing stop (initial stop applies before +1R).
  4. `partial_1R`: at +1R, realize 0.5 unit at +1R, move remainder (0.5 unit) stop → breakeven, remainder target = 3R. Trade `pnl = 0.5*R_banked_usd + 0.5*remainder_pnl_usd` (fractional units allowed in sim).
  5. `time_stop`: `fixed_1_5R` plus a hard exit at market on the first bar with ET time ≥ `TIME_STOP_ET="11:00"` if still open.
  - **Same-bar sequencing (Blocker 3 — no phantom credits):** on each managed bar, evaluate the current **pre-activation stop FIRST** using the base engine's stop-first / gap-through tie-break. If the bar's adverse extreme reaches the current stop, **exit as a stop** — do **not** allow +1R activation, breakeven-move, trailing, or a partial scale-out on that same bar. `+1R` activation (and any consequent breakeven/trail/partial) is permitted **only on bars that did not breach the current stop**. This keeps every mode conservative and prevents a bar that really stopped out at −1R from being re-credited as breakeven/partial/trail. `exit_reason ∈ {"stop","target","trail","time","partial_scaleout","partial_remainder_stop","partial_remainder_target"}`; each is costed by whether it is a market or limit fill (see cost model). `trail_swing`'s trailing level references only bars ≤ i (causal).
- **Volatility filter (leak-free + genuinely scale-free):** the filter variable is **ATR%** = `ATR14 / close * 100` at the signal bar — a **fraction of price**, NOT raw points, so it is invariant to the back-adjusted series' drifting price level (raw-point ATR would be confounded exactly like Phase-4's fvg%). `vol_filter ∈ {off, p25, p50, p75}`. The threshold is the p-th percentile of ATR% over a **single, pinned population**: **every in-session signal bar in the TRAIN window** (i.e. bars where a double-confirmation signal fired inside the session), computed **before** the vol gate and independent of the 1-trade/day cap and exit_mode — so the threshold is a stable property of the regime, not of realized fills, and both docs agree on this one definition. `off` = no gate. Enter only if the signal-bar `ATR% ≥ threshold`. **Leak-free:** the population is train-window-only; the OOS gate reads the causal `atr[i]` (bars ≤ i). The leak test must mutate train-window vs test-window ATR **separately** to catch any accidental global-percentile-over-full-df implementation.
- **Grid = 20:** `EXIT_MODES` (5) × `VOL_FILTERS` (4); entry params fixed at `StrategyParams()` defaults. `(fixed_1_5R, off)` = the base baseline; it MUST be in the grid.
- **Walk-forward:** reuse Phase-4 folds/precompute-slice/no-leakage machinery, with these Phase-5 overrides:
  - **NET metrics everywhere (Blocker 2):** the reused `_run_metrics`/`select_params` compute PF from **gross** `pnl_usd` — do NOT use them unchanged. A Phase-5 **net-aware metrics builder** computes PF/total/drawdown/expectancy from `t.net_pnl`; **selection, the 20-combo OOS null, the median-combo, condition-(d), and the stitched aggregates ALL use net.** Otherwise the "net verdict" is silently gross.
  - **ATR inside the layer (Blocker 4):** ATR% is added to the precomputed layer and sliced in `_slice_layer` with the identical `[a:b]` bounds as sig/ema/ohlc; `run_execution` reads the **sliced** `atr`, never a full-series array. Assert `len(atr) == len(sig)` at entry.
  - **Selection floor decoupled from the filter (Blocker 5):** `MIN_IS_TRADES=50` is evaluated on the **pre-filter in-session signal count** of the fold (which is identical across all 20 combos — it's a property of the entry base, not the exit/vol arm), so p50/p75 arms are NOT auto-disqualified by the very filter under test. Separately, a per-combo per-fold **eligibility + realized-trade table** is a required deliverable so the reader sees how thin the filtered arms get.
  - **Selection objective (net, in-sample):** max net PF among eligible combos; tie-break higher net total_pnl, then lower max_drawdown.
- **Pre-registered net success rule (falsifiable, small-sample-guarded):** positive iff ALL of —
  (a) tuned **net stitched-OOS PF > 1.0**;
  (b) tuned − base **net stitched-OOS PF ≥ 0.10**;
  (c) tuned beats base net PF in **≥ 3/4 folds**;
  (d) tuned net stitched-OOS PF **≥ the 75th percentile** of the 20 combos' **stitched** net-OOS PF (a stronger selection-luck null than ">median"; use the SAME stitched-PF statistic for the pick and the distribution — do not compare a stitched PF to a median-of-fold-medians);
  (e) **OOS-sample gate:** tuned **stitched-OOS trade count ≥ `MIN_OOS_TRADES = 60`** AND the tuned stitched-OOS net-PF **bootstrap CI lower bound (5th pct) > 1.0** (reuse `walkforward.bootstrap_pf_ci`, fixed seed). Without (e), the vol filter can shrink OOS n until a lucky low-n combo clears (a)–(d) by variance.
  Else → **null**. All conditions + values reported. Frozen; config hashed + git SHA; single-shot.
- **Config hash covers the WHOLE frozen design (Blocker-adjacent):** extend the Phase-4 `_config_hash` to include — the grid, fold windows, `MIN_IS_TRADES`, `MIN_OOS_TRADES`, the objective, the **vol-filter definition** (ATR% + pinned population + percentile set), the **ATR method** (period + Wilder/SMA), the **exit-mode intrabar sequencing**, the **cost constants**, and **`TIME_STOP_ET`** — so no frozen degree of freedom sits outside the hash.
- **`Trade` schema is EXTENDED ADDITIVELY:** append `net_pnl: float|None = None` and `exit_reason: str = ""` **after** existing fields; **keep `outcome`** and do not reorder — Phase-4's `_serialize_trade`/charts read `t.outcome`.
- **Reuse:** `metrics.py`; `tuning/walkforward.py` helpers (folds, `_slice_layer`, `select_params` generalized); the engine.

---

## Task 1: ATR + costs + exit modes (regression-locked)

**Files:** Create `strategy/atr.py`, `backtest/costs.py`, `backtest/exits.py`; Modify `strategy/params.py`, `backtest/engine.py`. Test: `tests/test_atr.py`, `tests/test_costs.py`, `tests/test_exits.py`, `tests/test_engine_p5_regression.py`.

**Interfaces:**
- Produces: `compute_atr(df, period=14) -> pd.Series` (causal); `CostModel(commission_rt, slippage_ticks_entry, slippage_ticks_stop, tick_value, multiplier=1.0)` + `apply_costs(gross, exit_reason) -> net`; the 5 exit-mode handlers in `exits.py`; `StrategyParams` with `exit_mode`, `vol_filter`; `run_execution(layer, params, fill_mode="next_open", cost_model=None, atr=None, vol_threshold=None) -> list[Trade]` (Trade gains `net_pnl`, `exit_reason`).

- [ ] **Step 1: `strategy/atr.py`.** `compute_atr(df, period=14)`: `TR = max(high-low, |high-prev_close|, |low-prev_close|)`; ATR = Wilder's RMA (or SMA of TR — pick RMA, document). Causal (uses bars ≤ i). Returns a Series aligned to df.index. Test `tests/test_atr.py`: TR/ATR values on a tiny hand frame; first value NaN handling; no look-ahead (a future bar can't change an earlier ATR).

- [ ] **Step 2: `backtest/costs.py`.** `CostModel` (frozen) with the pre-registered constants + `multiplier`. `MARKET_EXIT_REASONS = {"stop","trail","time","partial_remainder_stop"}`. `net_pnl(gross, exit_reason, n_fills=1)`: charges `COMMISSION_RT*n_fills` + `SLIPPAGE_TICKS_ENTRY*TICK_VALUE` (entry, once) + `SLIPPAGE_TICKS_EXIT*TICK_VALUE` for **each market-order exit fill** (0 for limit fills: `target`, `partial_scaleout`, `partial_remainder_target`), all × `multiplier`. For `partial_1R` the engine computes net **per leg** and sums (two commissions; scale-out leg = limit → 0 exit slippage; remainder leg costed by its own reason). Test `tests/test_costs.py`: target winner pays `COMMISSION_RT + 1 tick`; **trail/time/stop** exits each pay `COMMISSION_RT + 2 ticks` (entry + market exit) — i.e. trail/time are NOT under-costed as limits; multiplier 0 → net==gross; multiplier 2 → doubles; a partial pays two commissions with the correct per-leg slippage.

- [ ] **Step 3: `backtest/exits.py` (failing tests first).** Write `tests/test_exits.py` with a hand-built per-bar sequence for EACH mode asserting the exact exit price/reason:
  - `fixed_1_5R`: hits target at 1.5R → `("target", target_price)`; hits stop → `("stop", stop_price)`.
  - `breakeven_1R`: after a bar reaches +1R, a later bar returning to entry exits at entry as `("stop", entry)` with ~0 gross (before costs); if it never reaches +1R the original stop applies.
  - `trail_swing`: after +1R, price pulling back to the trailing swing exits `("trail", swing_level)`; a runner beyond 1.5R keeps running (no fixed-target cap).
  - `partial_1R`: reaching +1R banks 0.5 unit at +1R; remainder to 3R or breakeven; assert blended pnl for both remainder outcomes.
  - `time_stop`: still open at 11:00 ET → `("time", close_at_11:00)`.
  - **Intrabar stop-first (Blocker 3) — critical test:** a bar whose range spans BOTH the current stop and +1R (e.g. a wide bar that dips to the stop and also reaches +1R) must resolve as a **stop** in `breakeven_1R`, `trail_swing`, and `partial_1R` — NO breakeven move, trail, or scale-out credit on the bar that breached the stop. Assert the trade exits at the stop with a full −1R (before costs), not a phantom breakeven/partial.
  Then implement `exits.py` as a per-mode handler with explicit trade state (`activated_1R`, `stop_level`, `half_closed`, `remainder_stop`, `remainder_target`). **Order per managed bar: evaluate the current stop FIRST (stop-first / gap-through); only if the stop was not breached may +1R activate and consequent breakeven/trail/partial apply.**

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

- [ ] **Step 3: Implement `tuning/walkforward_p5.py`.** Reuse `make_folds`, `_precompute`, `_slice_layer` — but:
  - Add **ATR%** (`compute_atr(df)/df.close*100`) to the precomputed layer so `_slice_layer` cuts it with the identical `[a:b]` bounds (Blocker 4). Never pass a full-series ATR alongside a sliced layer.
  - Per fold: the vol-filter threshold = percentile of ATR% over **every in-session signal bar in the TRAIN slice** (the pinned population; `off` → None). Compute it from the train slice only.
  - Use a **net-aware metrics builder** (not the gross `_run_metrics`): run each of the 20 combos with the `cost_model`, compute PF/total/drawdown/expectancy from `t.net_pnl`.
  - `select_params` on **net** in-sample PF among **eligible** combos (eligibility = the fold's **pre-filter** in-session signal count ≥ `MIN_IS_TRADES`, identical across combos), tie-breaks per constraints.
  - OOS: selected + **all-20 combos' net OOS trades** (so each combo's **stitched** net-OOS PF can be computed across folds for condition (d)) + base `(fixed_1_5R, off)` net baseline. Cap net PF (inf) like Phase 4.
  - Return per-fold reports + **per-combo stitched net-OOS trade lists** (for the (d) distribution) + stitched net tuned/base + grid_size + the per-combo/per-fold eligibility & realized-trade counts.

- [ ] **Step 4: Run `pytest tests/test_grid_p5.py tests/test_walkforward_p5.py -q`.** Green. **Commit:** `feat: phase 5 grid + net walk-forward (leak-free ATR percentiles)`.

---

## Task 3: Runner + results.json + charts

**Files:** Create `run_phase5.py`; outputs `phase5_results.json` + `charts/phase5_*.png`. Test `tests/test_smoke_phase5.py`.

- [ ] **Step 1: `run_phase5.py` (single-shot).** `load_nq()` → `walk_forward_p5(df, build_grid_p5(), make_folds(), CostModel())`. `config_hash` = SHA-256 of the **whole frozen design** (grid + folds + MIN_IS_TRADES + MIN_OOS_TRADES + objective + vol-filter definition + ATR period/method + exit sequencing + cost constants + TIME_STOP_ET) + `git_sha`. Assemble `phase5_results.json`: config_hash, git_sha, grid_size(20), cost model, per-fold (selected exit_mode+vol_filter, net IS PF/n, net OOS metrics, **net OOS expectancy-per-trade**, base-net metrics, `selected_oos_percentile`, `median_combo_oos_pf`, `p75_combo_oos_pf`, fallback_used, and the **eligibility + realized-trade counts per combo**), stitched net OOS tuned vs base (all folds + excl. fallback), the **success_rule** — all **5** conditions (a)–(e) incl. the `MIN_OOS_TRADES=60` gate and the tuned stitched net-PF **bootstrap CI lower bound** — plus `robust_improvement`, and a **cost-sensitivity** block re-computing stitched tuned & base net PF/total_pnl at multipliers **0×/1×/2×** (does the verdict survive?). run_seconds.
- [ ] **Step 2: Charts (`charts/phase5_*.png`):** (1) stitched net OOS equity: tuned vs base vs gross; (2) per-fold net OOS PF tuned vs base; (3) selected exit_mode + vol_filter per fold (stability); (4) cost-sensitivity bars (net total_pnl at 0×/1×/2×); (5) 20-combo net-OOS-PF null with the selected pick + median marked.
- [ ] **Step 3: Run `python run_phase5.py` SYNCHRONOUSLY (foreground, wait ~5-15 min).** Commit `phase5_results.json` + PNGs.
- [ ] **Step 4: `tests/test_smoke_phase5.py`** — `walk_forward_p5` on a small grid + short real slice; structure + net metrics present; skip if raw data absent.
- [ ] **Step 5: `pytest tests/ -q` green.** **Commit:** `feat: phase 5 runner + results.json + charts`.

---

## Task 4: Notebook + writeup + final review/merge

- [ ] **Step 1: `WRITEUP_PHASE5.md`** — lead with the pre-registered NET verdict (`robust_improvement` yes/no) + all **5** conditions with numbers; the sub-story (which exit mode/vol filter got selected; did costs flip a gross-positive to net-negative; did the loss narrow like Phase 4). **Co-report expectancy-per-trade AND profit factor** (PF alone is a confound across heterogeneous exit shapes — `trail_swing` has unbounded upside while the others cap at 1.5R). Cost-sensitivity (does the verdict survive 0×→2×?). The selection-luck null + the tuned pick's OOS percentile + the eligibility/realized-trade table (how thin the filtered arms got). **Mandatory honesty disclosures:** (i) **cumulative multiple testing** — Phase 5 is the *third* pre-registered pass over the same 3-year back-adjusted series (after the Phase-2 build/validate and Phase-4's 144-combo sweep); (ii) this sweep is **conditional on the Phase-4-null default entry** (OOS PF ~0.99), so it asks "can exits+filter rescue a config already near-breakeven"; (iii) the `3R` remainder target, the `$5`/`1-tick` cost figures, and `TIME_STOP_ET=11:00` are **arbitrary pre-registered choices** (1-tick stop slippage is optimistic — the 2× band is the realistic case); (iv) n=4 folds is descriptive, overlapping windows inflate stability. If null, state it plainly. Program framing: Phase 5 is the last credible lever tried.
- [ ] **Step 2: `notebooks/05_costs_exits_volfilter.ipynb`** — load `phase5_results.json` (don't re-run the sweep), show fold table + 5 charts + the cost-sensitivity, narrate. Executes clean.
- [ ] **Step 3: Update `README.md`** — Phase 5 section + headline net verdict.
- [ ] **Step 4: Final whole-branch review** (superpowers:requesting-code-review over the branch diff); fix Critical/Important.
- [ ] **Step 5:** `pytest tests/ -q` green → merge `feat/phase5-exits-costs-volfilter` → master → vault update (`15-fyp-strategy-engine/_INDEX.md` Phase 5 + honest net result). **Do NOT push — held for the user.**

---

## Self-review notes
- **Spec coverage:** ATR+costs+exits regression-locked (T1), 20-combo net walk-forward + leak-free ATR% percentiles (T2), single-shot runner + 5-condition net success rule + cost sensitivity + null (T3), honest writeup+merge (T4).
- **Revised after a 3-lens adversarial review (DO-NOT-BUILD-as-written). All 5 blockers closed in-plan:** (1) **cost model** now charges slippage on every market exit (trail/time/partial-stop), not just `"stop"` — no differential under-costing of the treatment arm; (2) **net metrics everywhere** — selection/null/median/condition-(d)/stitched all use `net_pnl`, not the reused gross `_run_metrics`; (3) **intrabar stop-first** for all new modes — a bar that breaches the stop can't be re-credited as breakeven/trail/partial; (4) **ATR% lives inside the sliced layer** (identical `[a:b]` bounds, alignment assert) — no cross-window read; (5) **selection floor on the pre-filter signal count** so p50/p75 arms aren't auto-disqualified by the filter under test.
- **Important items folded in:** scale-free **ATR%** (not raw points) fixes the back-adjustment confound; **OOS-sample gate** (`MIN_OOS_TRADES=60` + bootstrap-CI-lower-bound>1) added to the success rule; condition (d) uses the consistent stitched-PF statistic at **p75**; **R anchored at signal-close**; `Trade` extended **additively** (keep `outcome`); **config hash covers the whole frozen design**; expectancy co-reported with PF; cumulative-multiple-testing + conditionality + arbitrary-choice disclosures mandated in the writeup.
- **Base-path regression invariants to preserve (final-review scrutiny):** gap-through stop fills at the WORSE of stop/open; stop-first same-bar tie-break; next-open fill; risk anchored at signal-close; loop start `range(swing, n)` — the `fixed_1_5R`/off/no-cost path must reproduce the Phase-2 golden fixture trade-for-trade.
- **Scrutinize in final review:** the intrabar stop-first ordering for partial/trail/breakeven; the ATR%-percentile-from-train-only leak guard (mutate train vs test ATR separately); net-metric routing; the per-leg partial cost/P&L; and that the config hash includes vol-filter/ATR/exit-sequencing/costs/TIME_STOP_ET.
