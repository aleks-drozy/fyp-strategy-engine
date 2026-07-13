# Phase 2 — Strategy Engine + Real-Log Validation

## Honest headline

A faithful, bar-by-bar Python reimplementation of the FYP IFVG+CISD NQ
strategy, run with the Pine script's **default** parameters over ~3 years of
1-minute NQ data, reproduces **76–80% of the real trade log's entry-days and
directions** (good recall) — but it also **over-generates roughly 4x as many
trades** (precision only 20–25%), and it **underperforms both real logs on
profit factor and win rate**.

| | 2023-24 log (losing) | Winning log |
|---|---|---|
| Real baseline (in-window, backtestable) | 95 trades / −$4,600 | 59 trades / +$18,115 |
| Generated, same window | 376 trades | 179 trades |
| Matched / Missed / Extra | 76 / 19 / 300 | 45 / 14 / 134 |
| Precision / Recall | 0.20 / 0.80 | 0.25 / 0.76 |
| Generated PF / WR / PnL | 0.71 / 33.8% / −$61,545 | 1.09 / 42.5% / +$11,067.50 |
| Real PF / WR / PnL | 0.90 / 37.9% / −$4,600 | 1.53 / 55.9% / +$18,115 |

This is not a flattering result, and it is reported as-is. The reimplementation
is **directionally honest**: on the losing 2023-24 period it correctly comes
out losing too (generated PF 0.71 vs real 0.90 — both below 1.0), and on the
winning period it correctly comes out winning too (generated PF 1.09 vs real
1.53 — both above 1.0). The engine gets the *sign* right on both regimes. What
it does not capture is the real track record's **selectivity**: the real logs
took roughly a quarter to a fifth as many trades as the raw default-parameter
signal fires on, and that selectivity is where most of the edge over the raw
signal lives.

**Interpretation:** the strategy's published "optimised" track record likely
relies on parameter tuning and/or additional discretionary filtering that
sits on top of the base IFVG+CISD double-confirmation logic — the base logic
alone, run honestly with no fitting to either log, does not reproduce the
real profit factor or win rate. This is not a failure of the rebuild; it is
the rebuild doing its job: isolating how much of the "optimised" edge comes
from the core strategy mechanic (validated: it fires on the right days, in
the right direction, most of the time) versus how much comes from tuning
(not validated: the default parameters alone over-trade and give back edge).
This finding is the direct motivation for **Phase 4** (parameter sweeps /
regime filter) — the next step is to search the parameter space the real
track record likely lived in, rather than assume the default Pine parameters
were ever the ones actually traded live.

## What was rebuilt

**Reused (ported, not rewritten) from `Trading_Dashboard-master/backend/strategy/`:**
- `strategy/ema.py` — `compute_ema`, a straight port of the Pine `ta.ema`
  logic via `pandas.ewm(adjust=False)`. Unchanged.
- `strategy/ifvg.py` — the Inversion Fair Value Gap state machine (gap
  detection → inversion → 10-bar expiry). Ported, then extended with the
  session gate (below).
- `strategy/cisd.py` — the Change-in-State-of-Delivery structure-break state
  machine. Ported, **with one confirmed bug fixed** (see below).

**Newly built for Phase 2:**
- `strategy/session.py` — the `09:30`–`10:30` America/New York session gate
  (left-inclusive/right-exclusive, weekdays only). The original port had no
  session concept; Pine gates both FVG creation and signal evaluation to this
  window, and the port did not.
- `strategy/signals.py` — the **double-confirmation transition trigger**
  (`double_confirmation`). This is the actual entry signal: it fires "Long"
  or "Short" only on the bar where CISD *flips* to a side while IFVG is
  already (or newly) aligned to that same side — not on every bar where the
  two happen to agree. This transition-only semantics did not exist in the
  original port and had to be built from the Pine source (lines 430–457)
  from scratch, with a truth-table test suite locking the four fire/no-fire
  cases.
- `backtest/engine.py` — the full execution layer: session-gated entries,
  the EMA close-side filter, the **8-bar inclusive swing stop**, the **1.5R
  target**, **1 trade/day**, always-flat (no pyramiding), fills at the
  **next bar's open** (`next_open`, fixed a-priori per Pine
  `strategy.entry` semantics — never selected to agree with the logs), and a
  **stop-first** same-bar resolution with gap-through fills at the worse of
  stop/open.
- `validate_trades.py` — parses the two real TradingView "List of Trades"
  CSV exports, window-clips both the generated and real sets to each log's
  own coverage window, and reports (entry-date, direction) coverage +
  per-side aggregate stats.

### The corrected CISD bug

The original port's structure-break "neighbor" index was off by one bar in
the wrong direction. Pine (line 195, bullish-break/max block) reads
`high[bar_index-bullishBreakIndex+1]`, which in Pine's "bars-ago" notation is
the neighbor **one bar earlier** than the break bar (`breakIdx-1`). The
original port computed `highs[i-offset+1]`, i.e. `highs[breakIdx+1]` — one
bar **later**, the wrong direction. This was corrected to
`highs[i-offset-1]` (`highs[breakIdx-1]`), with the mirrored fix applied to
the bearish-break/min block. `struct_top`/`struct_bottom` gate every
downstream structure break → CISD level → state flip → signal, so this one
line change affects every signal the engine produces. A characterization
test (`tests/test_cisd.py`) pins the corrected branch's output so a
regression back to the wrong neighbor fails loudly.

## Caveats (read before trusting any number above)

1. **The data and the real logs are different NQ continuations — prices are
   not directly comparable.** `nqdata.load_nq()` is a **back-adjusted
   continuous** NQ series (needed for clean multi-year backtesting across
   futures roll dates); the real trade logs are the **unadjusted
   front-month** contract. The offset between the two is large and
   time-varying (it shrinks toward the present as adjustment layers
   accumulate going back in time) — e.g. the matched-trade entry-price
   deltas run from roughly +2,655 points on 2023-era trades down to +256 on
   2025-era trades. **This means absolute entry/stop/target prices and
   dollar PnL are not directly comparable between generated and real
   trades**, and the multi-hundred/multi-thousand-point entry-price deltas
   reported in `backtest_results.json` (`matched_entry_price_delta`) are a
   **data-adjustment artifact, not a fill error**. Because the offset is
   (to first approximation) a per-day constant, it does not change
   intraday structure — gaps, EMA-relative position, stop distances — so it
   does not change *which* signals fire or their *win/loss outcome*. That
   is why the headline metrics here are **(entry-date, direction) coverage
   and win/loss outcomes**, which the offset does not affect, rather than
   raw price or $-PnL agreement.
2. **1-minute bars vs. the logs' tick-level fills.** The engine resolves
   entries/exits on 1-minute OHLC bars (stop-first, gap-through fills at the
   worse of stop/open); the real logs were produced against tick data. This
   is a second, independent source of divergence in exact fill prices, on
   top of caveat 1.
3. **The IFVG/CISD/EMA ports are unvalidated against TradingView itself.**
   There is no TradingView premium subscription available for this project
   (a founding constraint of the whole program), so the ported indicator
   logic could not be checked against a live Pine chart. Fidelity instead
   rests on (a) Pine-logic unit tests that pin specific state transitions
   against hand-derived expected sequences, and (b) this real-log
   validation — i.e., if the ported logic were badly wrong, it would be
   very unlikely to recover 76–80% date+direction recall against real
   trades it never saw. It is not proof of line-for-line Pine fidelity, but
   it is meaningful evidence against a badly broken port.
4. **`compute_ema` runs over the continuous ETH (electronic trading hours)
   series.** This only equals Pine's `ta.ema` exactly if the original Pine
   strategy's reference chart was also a continuous ETH series; if the
   Pine chart used RTH-only or a different session basis, the EMA filter
   will diverge slightly from the original, on top of caveats 1–2.

Taken together: the coverage/direction numbers are the trustworthy part of
this validation (offset-invariant, directionally consistent across both a
losing and a winning regime); the absolute PF/WR gap and all dollar figures
should be read as "the raw default-parameter strategy is a worse trader than
the real optimised track record," not as a precise dollar-for-dollar
reconciliation.

## Other honest bookkeeping

- **Out-of-coverage generated trades are reported, not hidden.** Of the 605
  generated trades, 50 fall outside both real logs' windows: 40 before the
  2023-24 log's first trade (2022-12-27 – 2023-03-03, before the real
  strategy's live track record starts), 10 in the gap between the two
  disjoint logs (2025-01-02 – 2025-01-20), and 0 after the data edge. These
  50 are excluded from both `compare()` reports by construction (window
  clip applies to both sides identically) and are surfaced separately in
  `backtest_results.json` under `generated.out_of_all_windows` so they don't
  silently vanish from any denominator.
- **The winning log's true backtestable baseline is 59 trades / +$18,115,
  not the full 72 / +$28,400.** 13 of the winning log's real trades (worth
  +$10,285, including some of its largest winners) fall after 2025-12-11,
  the last timestamp in the Phase-1 raw data. Comparing the generated engine
  against the full +$28,400 figure would be comparing against trades the
  data simply cannot produce a signal for — the in-window +$18,115 / 59
  trades is the only fair target.
- **0 same-bar-span exits.** No generated trade's exit bar hit both its stop
  and target on the same bar, so the pessimistic stop-first tie-break rule
  never actually had to arbitrate a genuine ambiguity in this run.

## Aggregate result (all 605 generated trades, unfiltered)

Profit factor **0.86**, win rate **37.0%**, total PnL **−$50,317.50**, max
drawdown **$72,617.50**, 312 Long / 293 Short. This is the full generated set
(including the 50 out-of-coverage trades) and is provided for completeness —
the in-window, per-log comparisons above are the ones that are actually
validated against a real baseline and should be treated as authoritative.
