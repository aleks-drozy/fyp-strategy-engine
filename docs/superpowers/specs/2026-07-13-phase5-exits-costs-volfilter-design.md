# Phase 5 — Costs + Exits + Volatility Filter — Design Spec

**Date:** 2026-07-13
**Owner:** Aleksandrs Drozdovs
**Status:** approved (brainstorming), pending spec review
**Program:** `fyp-strategy-engine` — Phase 5 (extension beyond the original 4-phase program)

## Purpose

Phases 2 & 4 established the strategy is ~breakeven-to-losing out-of-sample **gross**. Phase 5 asks the
honest follow-up: with **realistic trading costs** applied, can **smarter exits** and a **volatility filter**
turn it **net-profitable out-of-sample** — or does the last credible lever also come back null?

Everything is pre-registered and walk-forward, exactly like Phase 4. The headline bar is now **net of costs**,
which is a *higher* bar than Phase 4's gross numbers, not a lower one. The likely honest outcome is "modest
improvement, still not a reliable money-maker"; that is reported as-is.

## What this experiment isolates (and what it does NOT do)

- **Fixed base:** entry/signal parameters stay at the **Phase-2 defaults** (fvg 0, ema 20, swing 8, session
  09:30–10:30, 1 trade/day). Phase 5 does **not** re-open the entry param space — that would balloon the
  search and the overfitting risk. The experiment cleanly answers: *given the base signal, do better exits +
  a vol filter + honest costs change the verdict?*
- **Swept levers (the only new degrees of freedom):** `exit_mode` (5 values) × `vol_filter` (4 values) = **20
  combos**. Small and deliberate — far fewer hypotheses than Phase 4's 144.
- **Baseline to beat:** the base config = default entry + `fixed_1_5R` exit + `vol_filter=off`, **net of the
  same costs**. An improvement must beat THIS net baseline, not the gross default.

## Cost model (pre-registered constants)

NQ: tick = 0.25 pt, `$5/tick` (`$20/pt`). Applied per trade to produce `net_pnl`:
- `COMMISSION_RT = $5.00` (round-trip; retail NQ ~$2–2.50/side).
- `SLIPPAGE_TICKS_ENTRY = 1` — entry is a market/stop fill → 1 tick adverse.
- `SLIPPAGE_TICKS_STOP = 1` — stop exits are market → 1 tick adverse.
- Target exits are **limit** fills → **0** slippage.
- `net_pnl = gross_pnl − COMMISSION_RT − TICK_VALUE*(SLIPPAGE_TICKS_ENTRY + (SLIPPAGE_TICKS_STOP if exit_reason=="stop" else 0))`.
- **Sensitivity (reported, not for selection):** re-report the headline at cost multipliers **0×, 1× (headline), 2×** so the reader sees how cost-fragile the result is.

## Exit modes (pre-registered set of 5)

All keep the initial stop = N-bar swing (default 8) and initial risk R = |entry − stop|.
1. `fixed_1_5R` — **base:** target = entry ± 1.5R; stop fixed. (Phase-2/4 behavior.)
2. `breakeven_1R` — after price trades to +1R, move stop to entry (breakeven); target still 1.5R.
3. `trail_swing` — after +1R, trail the stop to the rolling N-bar swing; **no fixed target** (let winners run,
   exit only on the trailing stop). Before +1R, the initial stop applies.
4. `partial_1R` — bank half the position at +1R, move the remainder's stop to breakeven, remainder targets 3R
   (or trails swing if `trail`-style — pick one and fix it: remainder targets **3R**).
5. `time_stop` — `fixed_1_5R` PLUS a hard time exit: if still open at the session's end boundary (10:30 ET) +
   a fixed hold cap, close at market. Tests whether holding past the window hurts.

Exit fills obey the existing stop-first / gap-through model; intrabar ordering assumption unchanged.

## Volatility filter (pre-registered, leak-free, scale-free)

At the signal bar, compute `ATR14` = ATR over the trailing 14 one-minute bars. Filter: enter only if
`ATR14 ≥ threshold`, where the threshold is a **percentile of the IN-SAMPLE (train-window) distribution** of
signal-bar ATR14 — so it is scale-free and never uses test data. `vol_filter ∈ {off, p25, p50, p75}` (the
train-window 25th/50th/75th percentile; `off` = no filter). Percentiles are computed per fold from train
trades only.

## Walk-forward (reuse Phase 4 infra, verbatim discipline)

- Same 4 rolling folds (12mo train / 6mo test), same half-open boundaries, same precompute-and-slice,
  same leak-free selection, same selection-luck null — reuse `tuning/walkforward.py` machinery.
- **Selection objective (in-sample):** max **net-of-cost** profit factor, subject to `MIN_IS_TRADES = 50`;
  tie-break higher net total_pnl, then lower max_drawdown.
- **Null control:** run all 20 combos on each test slice; record the net-OOS-PF distribution, the selected
  pick's percentile, and the median-combo null.

## Pre-registered success rule (net of costs — the honest bar)

A **positive** ("costs+exits+filter make it robustly work") verdict requires ALL of:
(a) tuned **net** stitched-OOS PF **> 1.0** (actually profitable after costs);
(b) tuned net OOS PF exceeds the **base net** OOS PF by **≥ 0.10**;
(c) tuned beats base in **≥ 3/4 folds** (net);
(d) tuned net OOS PF **> the median of the 20 combos' net OOS PF** (selection-luck null).
Otherwise → **null** (with the honest sub-story: which levers helped, by how much, and whether it merely
narrowed the loss like Phase 4). Frozen before the run; config hashed + git-SHA'd; single-shot.

## Components (in `fyp-strategy-engine`)

```
strategy/atr.py            # compute_atr(df, period=14) -> pd.Series (causal, Wilder or SMA of TR)
backtest/costs.py          # CostModel dataclass + apply_costs(gross_pnl, exit_reason) -> net_pnl
backtest/exits.py          # exit-mode logic (the 5 modes) — a parameterized exit handler
backtest/engine.py         # thread exit_mode + vol filter (ATR gate) + costs through run_execution;
                           #   DEFAULT (fixed_1_5R, no filter, costs off) reproduces Phase-2/4 exactly
strategy/params.py         # extend StrategyParams: exit_mode="fixed_1_5R", vol_filter="off" (defaults
                           #   behavior-preserving); costs carried separately (CostModel), off by default
tuning/grid_p5.py          # 20-combo exit×vol grid (entry params fixed at defaults)
tuning/walkforward_p5.py   # thin wrapper: net-PF selection + ATR percentile thresholds per fold
run_phase5.py              # -> phase5_results.json + charts
notebooks/05_costs_exits_volfilter.ipynb ; WRITEUP_PHASE5.md
tests/ (atr, costs, each exit mode, vol-filter leak-free percentile, default-regression, walkforward)
```

## Testing (TDD)

- **Default-regression:** `run_execution` with `exit_mode="fixed_1_5R"`, `vol_filter="off"`, costs OFF
  reproduces the Phase-2 golden fixture trade-for-trade (parameterization is behavior-preserving).
- **Costs:** net_pnl math per trade; a winner (limit) pays commission + 1-tick entry slippage; a loser (stop)
  pays commission + entry + stop slippage; sensitivity multipliers scale correctly.
- **Each exit mode** on a hand-built frame realizes the intended behavior (breakeven can't lose after +1R
  reached; trail exits on the swing; partial banks half at +1R; time_stop closes at the boundary).
- **Vol filter is leak-free:** the percentile threshold is computed from train-window ATR only; a synthetic
  test proves a test-window ATR value never enters the threshold.
- **Walk-forward:** net-PF selection + all-20-combo null; no-leakage boundaries unchanged.

## Non-Goals

- **No** re-tuning of entry/signal params (fixed at defaults; joint entry+exit search is a future extension).
- **No** new instruments or new data (a separate robustness phase).
- **No** ML (the take/skip study already returned null).
- **No** tuning to the real logs.

## Risks

- **Overfitting** (exits are a classic curve-fit trap) → tiny grid (20), pre-registration, walk-forward,
  net-of-cost bar, selection-luck null.
- **Cost assumptions** → single pre-registered value + a 0×/1×/2× sensitivity band, disclosed.
- **Likely another null** → reported honestly; the sub-story (did exits narrow the loss? did the vol filter
  cut trade count usefully?) is the deliverable, as in Phases 2 & 4.
- **Intrabar exit realism** (trailing/partial need careful same-bar sequencing) → covered by explicit
  per-mode unit tests + the stop-first/gap-through assumption stated.
