# Phase 2 — Strategy Engine + Validation — Design Spec

**Date:** 2026-07-12
**Owner:** Aleksandrs Drozdovs
**Status:** approved (brainstorming), pending spec review
**Program:** `fyp-strategy-engine` — Phase 2 of 4 (Phase 1 data foundation is complete + public)

## Purpose

Reimplement the FYP **IFVG + CISD** NQ futures strategy as a **faithful, bar-by-bar Python
backtester** over the Phase-1 NQ 1-minute data, then **validate the generated trades against the two
real trade logs**. The goal is a trustworthy trade generator (feeds Phase 3 scale-analysis and Phase 4
improvement experiments) plus an honest, portfolio-grade write-up of how closely a free-data Python
rebuild can reproduce a TradingView strategy.

This is a **faithful approximation, not a clone** (see Honesty Caveats). Success is judged by honestly
measured overlap and comparable aggregate stats on the shared date window — never by cherry-picking.

## Authoritative source

The strict spec is the Pine strategy `FYP_BOT_1_3.pine` (542 lines) at
`C:/Users/Alex/Projects/Trading_Dashboard/Trading_Dashboard-master/docs/reference/`. Every rule below is
taken from that file (line numbers cited). Where prior vault notes disagreed, **the Pine wins**.

### Exact rules (from the Pine)

- **Inputs:** `maxTradesPerDay = 1` (line 7), `swingLookback = 8` (line 8), `riskRewardRatio = 1.5`
  (line 9), EMA `emaLength = 20` on `close` (lines 495-499), `fvgThreshold = 0%` (line 293, so any
  gap qualifies), `endMethod = "Close"` (line 294, invalidation on close), `ifvgLookback = 10` (line 462).
- **Session (lines 13-33):** 09:30–10:30 in the chart/exchange timezone (`startHour=9,startMinute=30,
  endHour=10,endMinute=30`), `inSession = t >= 09:30 and t < 10:30` (left-inclusive, right-exclusive),
  **weekdays only**. For NQ the exchange tz is New York, so this is **09:30–10:30 America/New_York**.
- **IFVG (lines 292-484):** bullish gap `low > high[2]`, bearish gap `high < low[2]` (3-bar gaps).
  Gaps are **created only when `inTradingSession`** (lines 320, 335). Bullish FVG stores
  `top=low, bottom=high[2]`; bearish stores `top=low[2], bottom=high`. Inversion (endMethod="Close"):
  bullish FVG inverts when `close < bottom`; bearish when `close > top`. An **inverted bullish** FVG →
  `ifvgState="Bearish"`; an **inverted bearish** FVG → `ifvgState="Bullish"` (line 420). `fvgArray` is
  **cleared each calendar day** (line 134); `ifvgState` resets to `"None"` outside session (lines 425-426).
  The IFVG entry level (`ifvgTop`/`ifvgBottom`) is only valid while `bar_index - invertBar <= 10` (lines
  472-484) — this 10-bar expiry gates entries via `not na(ifvgBottom)` / `not na(ifvgTop)`.
- **CISD (lines 37-289):** a market-structure state machine — tracks bullish/bearish pullbacks and
  structure breaks, maintains one active bull/bear CISD level (keepLevels=false), and flips a boolean
  `currentState` (True=Bullish) when `close` crosses the opposite CISD level (lines 280-288).
- **Double-confirmation entry trigger (lines 430-457, 508-509) — the crux:**
  the signal fires on a **transition bar**, not on static alignment:
  - `cisdTurnedBullish = (prev currentState == False) and (cur currentState == True)`
  - `ifvgTurnedBullish = (prev ifvgState != "Bullish") and (cur ifvgState == "Bullish")`
  - `bullDouble = (prev ifvg=="Bullish" and cur ifvg=="Bullish" and cisdTurnedBullish) or (ifvgTurnedBullish and cisdTurnedBullish)`
  - (bearish is the mirror). In plain terms: **on the bar the CISD flips, if the IFVG is (already or
    newly) on the same side, the signal fires.** This transition logic is the piece **no existing Python
    has implemented** — the current dashboard backtest fires on static "both aligned", which is wrong.
- **EMA trend filter (lines 508-509):** long requires `close > ema20`; short requires `close < ema20`.
- **Entry execution (lines 504-529):** enter only when flat (`position_size==0`), in session,
  `tradesToday < 1`, filter passed, and a non-expired IFVG level exists. Long stop `= ta.lowest(low, 8)`
  (lowest low of the last 8 bars **including the current bar**); short stop `= ta.highest(high, 8)`.
  `risk = |close - stop|`; target `= close ± risk * 1.5`. **Fixed 1 contract.** Exit via stop or limit.

## Reuse — what already exists (major accelerator)

A faithful, line-by-line Python port of the hardest pieces lives in the Trading Dashboard repo and is
**directly reusable** over `load_nq()` output (same lowercase OHLCV contract):

- `backend/strategy/ifvg.py` — `compute_ifvg(df)` state machine. **HIGH reuse.** **Change required:** it
  omits the `inTradingSession` gate on FVG creation (it was built for a context where session was handled
  elsewhere). Phase 2 must add the session gate so only in-session FVGs exist, matching the Pine.
- `backend/strategy/cisd.py` — `compute_cisd(df)` (~200 lines). **HIGH reuse, drop-in.**
- `backend/strategy/ema.py` — `compute_ema(df, 20)` using `ewm(span=20, adjust=False)` (matches
  TradingView's recursive EMA — the load-bearing correctness detail). **Reuse verbatim.**
- `backend/strategy/engine.py` — the `df.iloc[:-1]` lookahead guardrail pattern. **Copy the pattern.**
- `backend/backtest/router.py` — the bar-by-bar exit-simulation loop (stop/target hit detection, equity
  curve). **Reuse the loop shape**, but **replace its entry rule** (it uses the simplified static-alignment
  rule and lacks the session gate + 1/day cap + fixed-contract sizing).
- Test fixtures under `Trading_Dashboard-master/backend/**/fixtures` and `.../data/fixtures` (encode Pine
  output) should be reused where present to lock the ports; otherwise synthetic fixtures.
- From the Monte Carlo repo: `mc/metrics.py` (`profit_factor`, `total_pnl`, `win_rate`, `max_drawdown`)
  and `data/schema.py` (TradingView-CSV adapter) — reuse for metrics and for parsing the real logs.

**Provenance:** ported modules keep a docstring line citing the Pine and the original path. The ports were
**never validated against TradingView ground truth** (noted in the dashboard's own docs) — so the newly
added session gate and the double-confirmation logic get **extra test scrutiny**.

## Components (built in `fyp-strategy-engine`)

```
fyp-strategy-engine/
  strategy/
    __init__.py
    ema.py        # compute_ema(df, period=20)         [ported verbatim]
    ifvg.py       # compute_ifvg(df, in_session)        [ported + session gate added]
    cisd.py       # compute_cisd(df)                     [ported]
    session.py    # in_session_mask(index, start="09:30", end="10:30")  -> bool Series (NY tz, weekday)
    signals.py    # double_confirmation(ifvg, cisd) -> Series of {"", "Long", "Short"} transition signals
  backtest/
    __init__.py
    trade.py      # Trade dataclass (entry_time, direction, entry, stop, target, exit_time, exit, pnl_usd, r, outcome)
    engine.py     # backtest(df) -> list[Trade]: session-gated entries, EMA filter, 8-bar stop, 1.5R,
                  #   max 1/day, flat-only, stop-first exit sim, 1 contract ($20/pt)
  validate_trades.py   # parse real logs + join generated<->real on (entry_date, direction); report
  run_backtest.py      # backtest over load_nq() -> backtest_results.json + charts
  notebooks/03_strategy_engine.ipynb
  WRITEUP_STRATEGY.md
  tests/  (test_session, test_signals, test_ifvg_gate, test_backtest_engine, test_validate_trades, ...)
```

## Validation methodology

**Match key:** entry **session-date + direction**. Verified unique in both logs (one trade/day), so this
is unambiguous; direction is a guard so a right-day/wrong-side trade scores as a **mismatch**, not a match.

**Data window:** the engine runs over `load_nq()` (2022-12-26 → **2025-12-11**). Real trades are compared
**only within that window**: real trades dated after 2025-12-11 are **excluded as out-of-coverage** (not
counted as misses), and the count excluded is reported explicitly.

- **`NQ1_2023_2024.csv`** (95 trades, −$4,600, PF 0.90) — **entirely inside** the window → full check.
- **`NQ1_optimised.csv`** (72 trades, +$28,400, PF 1.70) — runs Jan 2025 → Feb 2026; the tail after
  2025-12-11 is excluded (exact count computed and reported).

**Report (`backtest_results.json` + write-up):**
- Coverage: # real trades in window, # excluded (post-truncation), # generated, # matched (same
  date+direction), # real-only (missed), # generated-only (extra).
- Matched-trade diagnostics: entry-price delta distribution, exit-outcome agreement (win/loss), PnL
  correlation. Entry-price exactness is a **diagnostic, not a pass/fail gate** (1-min bars vs tick fills).
- Aggregate side-by-side on the shared window: profit factor, win rate, total PnL, direction mix —
  **generated vs real**, per log.

**Faithfulness bar (honest, achievable):** aim to reproduce a **substantial majority of real trade-days
with correct direction**, with aggregate PF/WR in the **same ballpark** on the shared window. No hard
pass/fail percentage is hard-coded; the measured overlap is reported plainly, including when it is low.
Reproducing the *losing* 2023-24 log's character matters as much as the winner — that is the honesty test.

## Modeling decisions (locked, with rationale)

- **Fill convention:** entry executes on the bar **after** the signal at its **open** (Pine
  `strategy.entry` default), with stop/target **anchored to the signal-bar close** (as the Pine computes
  them). A `fill_mode` flag also supports signal-bar-close fills; validation picks whichever matches the
  real logs' entry prices better, and the choice is documented. Entry-price deltas are reported either way.
- **Swing stop window:** `ta.lowest(low, 8)` / `ta.highest(high, 8)` include the **current bar** (8 bars
  ending at the entry bar, inclusive). This differs from the dashboard router's `[i-8:i]` (exclusive) —
  Phase 2 matches the Pine (inclusive).
- **Same-bar stop-and-target:** if a bar's range spans both, assume **stop first** (pessimistic/honest).
- **Exit timing:** exits are **not** session-gated — a position opened in-session exits whenever stop/TP
  hits, even hours later (matches the real logs, which hold past 10:30). No time-based exit exists.
- **Sizing:** fixed **1 contract**; PnL = `(exit − entry) × direction × $20/pt`, gross (no
  commission/slippage) — matches how both real logs are recorded (verified: log PnL == price-delta × 20).

## Testing (TDD)

- `session.py`: 09:30 included, 10:30 excluded, weekend excluded, DST-correct (NY wall-clock).
- `signals.py`: truth-table tests of `double_confirmation` on hand-built (ifvg, cisd) sequences — fires
  only on the transition bar, both bull and bear; no fire on static alignment.
- `ifvg.py` session gate: an FVG whose formation bar is outside session is never created; in-session
  formation behaves as before. Lock the base port on a fixture (reuse dashboard fixtures if available).
- `backtest/engine.py`: 8-bar inclusive stop, 1.5R target math, stop-first same-bar exit, max-1-trade/day,
  flat-only (no pyramiding), no-lookahead (entry decided on closed bars only), out-of-session no entry.
- `validate_trades.py`: log parsing (entry/exit rows pair correctly; PnL identity holds), the
  (date,direction) join, and the coverage/exclusion accounting on a synthetic set.
- Real-data smoke: run the engine on a small real slice; assert plausible trades + no exceptions.

## Non-Goals

- **No** Phase 3 (large-sample Monte Carlo / ML re-run) — Phase 2 stops at a validated engine + report.
- **No** Phase 4 improvement experiments (parameter sweeps, regime filter).
- **No** topping up the Dec-2025→Feb-2026 data gap (Dukascopy option deferred to a later phase).
- **No** attempt to reproduce tick-level fills — 1-minute bars are the accepted resolution.

## Risks

- **Ports never validated vs TradingView** → the added session gate + double-confirmation logic carry the
  most fidelity risk; mitigate with focused unit tests and by validating against BOTH real logs.
- **1-min vs tick fills** → exact entry/exit prices will differ; mitigated by keying validation on
  date+direction and treating price deltas as diagnostics.
- **Dec-2025 truncation** → the winning log's tail is un-backtestable; handled by explicit exclusion +
  reporting, never silent.
- **Faithfulness may be modest** → if overlap is low, that is reported honestly as the finding; the
  portfolio value is the rigorous method, exactly as with the Monte Carlo and ML studies.
