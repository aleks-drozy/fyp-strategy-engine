# Phase 2 — Strategy Engine + Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully reimplement the FYP IFVG+CISD NQ strategy as a bar-by-bar Python backtester over the Phase-1 NQ data, and validate the generated trades against the two real trade logs on (entry-date, direction).

**Architecture:** Reuse the existing faithful IFVG/CISD/EMA state-machine ports; add the missing session gate and the true double-confirmation *transition* trigger; build a correct execution layer (session-gated entries, EMA filter, 8-bar inclusive swing stop, 1.5R target, 1 trade/day, 1 contract, stop-first exit sim); then a validator that joins generated↔real trades and reports coverage + aggregate stats honestly.

**Tech Stack:** Python, pandas 2.2.3, numpy 2.1.3, pytest 8.3.3, matplotlib (charts). Pure functions over `nqdata.load_nq()` output.

## Global Constraints

- **Data contract:** all indicator/engine functions take a DataFrame with lowercase columns `open, high, low, close, volume` and a **tz-aware `US/Eastern` DatetimeIndex** (exactly `nqdata.load_nq()` output). Prices are on the **0.25** grid.
- **Session:** `09:30`–`10:30` **America/New_York**, **weekdays only**, left-inclusive / right-exclusive (`t >= 09:30 and t < 10:30`). This is the strategy's own window — do **not** use the Phase-1 `session_slice` default of 09:32–10:00.
- **Strategy params (from `FYP_BOT_1_3.pine`):** `swingLookback=8` (inclusive of current bar), `riskRewardRatio=1.5`, `emaLength=20` (`ewm(span=20, adjust=False)`), `maxTradesPerDay=1`, `fvgThreshold=0` (any gap), `ifvgLookback=10` (IFVG expiry), `endMethod="Close"`.
- **Contract:** NQ = **$20 / index point**. PnL is gross (no commission/slippage): `(exit − entry) × (+1 long / −1 short) × 20`.
- **No lookahead:** every decision at bar `i` uses only bars `≤ i` (closed). Entry fills on the bar **after** the signal (fill at its open); stop/target anchored to the **signal-bar close**.
- **Fill mode is fixed a-priori** to `next_open` per Pine `strategy.entry` semantics — it is **never selected by agreement with the real logs** (that would curve-fit the reimplementation to the validation set). `signal_close` fills may be reported only as a **pre-declared sensitivity band**, never to choose the headline.
- **`MAX_TRADES_PER_DAY = 1`** as a named constant (not a hardcoded literal), mirroring Pine `maxTradesPerDay`.
- **DATA vs LOGS ARE DIFFERENT NQ CONTINUATIONS — not directly price-comparable.** `load_nq` is a **back-adjusted continuous** series; the real logs are the **unadjusted front-month**. Verified: load_nq 15017 vs real 12362 (2023-03-06), 22523 vs 21610 (2025-01-21) — a large, time-varying offset that shrinks toward the present. A per-day-constant offset preserves all intraday relationships (gaps, structure, EMA-relative, stop distances under additive adjustment), so **signals are unaffected**, but **absolute entry/stop/target prices must NOT be compared raw** to the logs. **Headline validation metrics are (entry-date, direction) coverage and win/loss outcomes** (offset-invariant); absolute-price and $-PnL comparisons are secondary and must be offset-corrected or explicitly caveated as a data-adjustment artifact — never attributed to fill granularity.
- **Match key (validation):** entry **NY session-date + direction** (verified unique per day in both logs). Compare only within the data window, applying **both bounds** `2022-12-26 ≤ entry-date ≤ 2025-12-11`; real trades outside the window are **excluded and counted** (never scored as misses), and the **generated set is filtered to the SAME window as each log** before comparing (the two logs are disjoint: 2023-24 vs 2025-26).
- **True in-window baselines (verified — use these, not the full-log figures):** 2023-24 log = **95 trades / −$4,600** (fully in-window); winning log = **59 trades / +$18,115** in-window (the 13 post-2025-12-11 trades carrying +$10,285, incl. the biggest winners, are out-of-coverage). The full winning-log +$28,400/72 is **not** the backtestable target.
- **Provenance:** ported modules keep a docstring line citing `FYP_BOT_1_3.pine` and the original `Trading_Dashboard-master/backend/strategy/<f>.py` path. Ports are **unvalidated against TradingView** (the reference fixtures are empty placeholders and we have no TV premium), so fidelity is checked via **synthetic Pine-logic unit tests** and the **real-log validation** — not against TV output.
- **Sources (read-only, copy from):**
  - Pine: `C:/Users/Alex/Projects/Trading_Dashboard/Trading_Dashboard-master/docs/reference/FYP_BOT_1_3.pine`
  - Ports: `.../backend/strategy/{ema,ifvg,cisd}.py`; exit-loop reference `.../backend/backtest/router.py`
  - Real logs: `C:/Users/Alex/Projects/Trading-Strategy-Monte-Carlo-Simulation/data/{NQ1_optimised.csv,NQ1_2023_2024.csv}`
  - Metrics to reuse: `.../Trading-Strategy-Monte-Carlo-Simulation/mc/metrics.py` (`profit_factor,total_pnl,win_rate,max_drawdown`).

---

## Task 1: Strategy ports + session gate

**Files:**
- Create: `strategy/__init__.py`, `strategy/ema.py`, `strategy/session.py`, `strategy/ifvg.py`, `strategy/cisd.py`
- Test: `tests/test_session.py`, `tests/test_ifvg.py`, `tests/test_cisd.py`, `tests/test_ema.py`
- Modify: `requirements.txt` (add `matplotlib==3.9.2`)

**Interfaces:**
- Produces: `compute_ema(df, period=20) -> pd.Series`; `in_session_mask(index, start="09:30", end="10:30") -> pd.Series[bool]`; `compute_ifvg(df, in_session: pd.Series) -> pd.Series` (values `"Bullish"/"Bearish"/"None"/"Expired"`); `compute_cisd(df) -> pd.Series` (`"Bullish"/"Bearish"`).

- [ ] **Step 1: Copy `ema.py`; copy `cisd.py` and FIX its off-by-one.** Copy `.../backend/strategy/ema.py` → `strategy/ema.py` unchanged (add a provenance docstring line). Copy `.../backend/strategy/cisd.py` → `strategy/cisd.py` but **correct a confirmed sign error** vs the Pine (do NOT trust the port as line-faithful — this branch is wrong):
  - **Pine line 195** uses `math.max(high[bar_index-bullishBreakIndex], high[bar_index-bullishBreakIndex+1])` → the bars `{breakIdx, breakIdx-1}` (bars-ago index: `+1` means one bar *earlier*). The port (lines 113-116) computes `offset=i-bullish_break_idx; h1=highs[i-offset]=highs[breakIdx]; h2=highs[i-offset+1]=highs[breakIdx+1]` → bars `{breakIdx, breakIdx+1}` (one bar *later*, wrong direction).
  - **Fix (bullish-break / max block):** `h2 = highs[i-offset-1]` (i.e. `highs[breakIdx-1]`), guarding `i-offset-1 >= 0`.
  - **Mirror fix (bearish-break / min block, Pine line 217):** `l2 = lows[i-offset-1]` (i.e. `lows[breakIdx-1]`), guarding index `>= 0`.
  - Add a provenance docstring line noting the corrected neighbor index. This `struct_top/struct_bottom` gates every future structure break → CISD level → `currentState` → signal, so it must be right. Step 8 adds a test locking this branch.
  - Remove any `backend.` import prefixes (none in these two).

- [ ] **Step 2: Write `strategy/session.py`.**

```python
"""NY trading-session mask. Strategy window = 09:30-10:30 America/New_York, weekdays,
left-inclusive/right-exclusive. From FYP_BOT_1_3.pine lines 13-33."""
import pandas as pd

def in_session_mask(index: pd.DatetimeIndex, start: str = "09:30", end: str = "10:30") -> pd.Series:
    if index.tz is None:
        raise ValueError("index must be tz-aware (US/Eastern)")
    ny = index.tz_convert("America/New_York")
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    mins = ny.hour * 60 + ny.minute
    in_win = (mins >= sh * 60 + sm) & (mins < eh * 60 + em)
    is_weekday = ny.dayofweek < 5  # Mon..Fri
    return pd.Series(in_win & is_weekday, index=index, name="in_session")
```

- [ ] **Step 3: Write `tests/test_session.py` (failing).**

```python
import pandas as pd
from strategy.session import in_session_mask

def _idx(times):  # times: list of "YYYY-MM-DD HH:MM"
    return pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("US/Eastern")

def test_bounds_and_weekday():
    idx = _idx(["2025-01-21 09:29","2025-01-21 09:30","2025-01-21 10:29",
                "2025-01-21 10:30","2025-01-25 09:45"])  # last is a Saturday
    m = in_session_mask(idx).tolist()
    assert m == [False, True, True, False, False]

def test_requires_tz():
    import pytest
    with pytest.raises(ValueError):
        in_session_mask(pd.DatetimeIndex(pd.to_datetime(["2025-01-21 09:30"])))
```

- [ ] **Step 4: Run `pytest tests/test_session.py -q`.** Expected: PASS (2 tests).

- [ ] **Step 5: Copy `ifvg.py` and add the session gate.** Copy `.../backend/strategy/ifvg.py` → `strategy/ifvg.py`. Change the signature to `compute_ifvg(df: pd.DataFrame, in_session: pd.Series) -> pd.Series` and apply exactly these edits (mirrors Pine lines 320/335 gating FVG creation, and lines 425-426 resetting state outside session):
  - Before the loop: `session = in_session.to_numpy(dtype=bool)`.
  - Gate **both** FVG-creation blocks on session: wrap `if bullish_gap:` as `if bullish_gap and session[i]:` and `if bearish_gap:` as `if bearish_gap and session[i]:`.
  - After computing `ifvg_state` for bar `i`, add: `if not session[i]: ifvg_state = "None"`; then `states[i] = ifvg_state`.
  - Keep the daily calendar reset and the 10-bar expiry logic unchanged. Update the docstring to note the added `in_session` gate.

- [ ] **Step 6: Write `tests/test_ifvg.py` (synthetic Pine-logic, failing).** Build a tiny deterministic frame (index all inside session on a weekday) that forms a bullish FVG then inverts, and assert the state transitions; add a case where the gap-forming bar is out of session and assert **no** FVG (state stays `"None"`).

```python
import pandas as pd
from strategy.ifvg import compute_ifvg
from strategy.session import in_session_mask

def _frame(rows, day="2025-01-21", t0=" 09:32"):
    # rows: list of (open,high,low,close); consecutive 1-min bars
    idx = pd.date_range(f"{day}{t0}", periods=len(rows), freq="1min", tz="US/Eastern")
    df = pd.DataFrame(rows, columns=["open","high","low","close"], index=idx)
    df["volume"] = 1
    return df

def test_bullish_fvg_forms_and_inverts_in_session():
    # bar2 low(105) > bar0 high(100) => bullish FVG {top=105, bottom=100};
    # later a close below 100 inverts it -> ifvgState "Bearish"
    rows = [(99,100,98,99.5),(101,104,100,103),(106,108,105,107),
            (104,104,101,102),(101,101,99,99.0)]
    df = _frame(rows)
    st = compute_ifvg(df, in_session_mask(df.index))
    assert st.iloc[2] in ("None","Bullish")   # FVG created, not yet inverted
    assert st.iloc[4] == "Bearish"             # inverted bullish FVG -> Bearish signal

def test_no_fvg_created_out_of_session():
    df = _frame([(99,100,98,99.5),(101,104,100,103),(106,108,105,107)],
                t0=" 08:00")  # pre-session
    st = compute_ifvg(df, in_session_mask(df.index))
    assert set(st.unique()) <= {"None"}
```

- [ ] **Step 7: Run `pytest tests/test_ifvg.py -q`.** Expected: PASS. (Adjust the synthetic OHLC only if needed to realize the intended gap/inversion; keep the assertions.)

- [ ] **Step 8: Write `tests/test_cisd.py` and `tests/test_ema.py`.** For CISD: (a) smoke — feed ~30 synthetic bars, assert a `"Bullish"/"Bearish"` Series of matching length, no NaN; (b) **characterization test locking the corrected pullback-break branch** — a fixed, hand-built sequence that drives a bearish pullback followed by a bullish structure break (so the corrected `highs[breakIdx-1]` neighbor is exercised), asserting the **full output state series equals a pinned expected list**. Add a comment: this pins the corrected neighbor index (Blocker 4); a revert to `breakIdx+1` changes the pinned output and fails. For EMA: assert `compute_ema` equals `df["close"].ewm(span=20, adjust=False).mean()`; **drop the false "first period-1 values are NaN" docstring claim** (ewm adjust=False seeds from bar 0, no NaN) and note the seed-warmup difference vs Pine `ta.ema` is negligible after session warmup.

- [ ] **Step 9: Run `pytest tests/ -q`.** Expected: all green. **Commit:** `feat: port ema/cisd/ifvg + session gate into strategy/`.

---

## Task 2: Double-confirmation transition signals

**Files:** Create `strategy/signals.py`; Test `tests/test_signals.py`.

**Interfaces:**
- Consumes: `compute_ifvg` output, `compute_cisd` output (aligned Series).
- Produces: `double_confirmation(ifvg: pd.Series, cisd: pd.Series) -> pd.Series` returning `""`/`"Long"`/`"Short"` per bar — non-empty **only on the transition bar**, exactly Pine `bullDouble`/`bearDouble` (lines 430-457).

- [ ] **Step 1: Write `tests/test_signals.py` (failing).** Truth-table on hand-built state sequences:

```python
import pandas as pd
from strategy.signals import double_confirmation

def _s(vals): return pd.Series(vals)

def test_long_fires_only_on_cisd_flip_bar_when_ifvg_bullish():
    # bar2: cisd flips Bearish->Bullish while ifvg already Bullish => Long on bar2 only
    ifvg = _s(["Bullish","Bullish","Bullish","Bullish"])
    cisd = _s(["Bearish","Bearish","Bullish","Bullish"])
    out = double_confirmation(ifvg, cisd).tolist()
    assert out == ["", "", "Long", ""]

def test_no_fire_on_static_alignment():
    ifvg = _s(["Bullish","Bullish","Bullish"])
    cisd = _s(["Bullish","Bullish","Bullish"])   # already aligned, no flip
    assert double_confirmation(ifvg, cisd).tolist() == ["","",""]

def test_short_mirror():
    ifvg = _s(["Bearish","Bearish","Bearish"])
    cisd = _s(["Bullish","Bullish","Bearish"])   # flip to Bearish on bar2
    assert double_confirmation(ifvg, cisd).tolist() == ["","","Short"]

def test_ifvg_turns_and_cisd_flips_same_bar():
    ifvg = _s(["None","None","Bullish"])
    cisd = _s(["Bearish","Bearish","Bullish"])   # both turn bullish on bar2
    assert double_confirmation(ifvg, cisd).tolist() == ["","","Long"]
```

- [ ] **Step 2: Run `pytest tests/test_signals.py -q`.** Expected: FAIL (module missing).

- [ ] **Step 3: Implement `strategy/signals.py`.**

```python
"""Double-confirmation entry trigger. From FYP_BOT_1_3.pine lines 430-457, 508-509.
Fires on the bar where CISD flips, if the IFVG is (already or newly) on the same side."""
import pandas as pd

def double_confirmation(ifvg: pd.Series, cisd: pd.Series) -> pd.Series:
    ifvg = list(ifvg); cisd = list(cisd)
    n = len(ifvg)
    cisd_bull = [c == "Bullish" for c in cisd]
    out = [""] * n
    for i in range(1, n):
        cisd_turned_bull = (not cisd_bull[i-1]) and cisd_bull[i]
        cisd_turned_bear = cisd_bull[i-1] and (not cisd_bull[i])
        ifvg_turned_bull = (ifvg[i-1] != "Bullish") and (ifvg[i] == "Bullish")
        ifvg_turned_bear = (ifvg[i-1] != "Bearish") and (ifvg[i] == "Bearish")
        bull_double = ((ifvg[i-1] == "Bullish" and ifvg[i] == "Bullish" and cisd_turned_bull)
                       or (ifvg_turned_bull and cisd_turned_bull))
        bear_double = ((ifvg[i-1] == "Bearish" and ifvg[i] == "Bearish" and cisd_turned_bear)
                       or (ifvg_turned_bear and cisd_turned_bear))
        if bull_double:
            out[i] = "Long"
        elif bear_double:
            out[i] = "Short"
    return pd.Series(out, index=ifvg_index(ifvg, cisd), name="signal")

def ifvg_index(ifvg, cisd):  # helper kept simple; caller passes Series, we reindex in engine
    return range(len(ifvg))
```
  Note: the engine will pass Series and re-attach the DatetimeIndex; keep `double_confirmation` index-agnostic (positional). If simpler, return `pd.Series(out)` and let the engine set the index — update the tests to compare `.tolist()` (already the case).

- [ ] **Step 4: Run `pytest tests/test_signals.py -q`.** Expected: PASS (4 tests). **Commit:** `feat: double-confirmation transition signal`.

---

## Task 3: Backtest engine

**Files:** Create `backtest/__init__.py`, `backtest/trade.py`, `backtest/engine.py`; Test `tests/test_engine.py`.

**Interfaces:**
- Consumes: `strategy.*` (Task 1-2), `nqdata.load_nq` shape.
- Produces: `@dataclass Trade(entry_time, direction, entry, stop, target, exit_time, exit, pnl_usd, r_multiple, outcome)`; `backtest(df, fill_mode="next_open") -> list[Trade]`.

- [ ] **Step 1: Write `backtest/trade.py`.**

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    entry_time: datetime
    direction: str        # "Long" | "Short"
    entry: float
    stop: float
    target: float
    exit_time: datetime | None = None
    exit: float | None = None
    pnl_usd: float | None = None
    r_multiple: float | None = None
    outcome: str = "Open"  # "Win" | "Loss" | "Open"
```

- [ ] **Step 2: Write `tests/test_engine.py` (failing).** Cover: (a) a hand-built session frame that produces exactly one long trade hitting target → `outcome=="Win"`, `pnl_usd == (target-entry)*20`; (b) stop-first when a bar spans both; (c) max-1-trade/day (two signals same day → one trade); (d) no entry outside session; (e) no-lookahead (a spike on a future bar cannot change an earlier entry). Use small synthetic frames; assert on the returned `Trade` list.

- [ ] **Step 3: Run `pytest tests/test_engine.py -q`.** Expected: FAIL.

- [ ] **Step 4: Implement `backtest/engine.py`.** Core algorithm (bar loop; `$20/pt`; `SWING=8`, `RR=1.5`):

```python
import numpy as np, pandas as pd
from strategy.session import in_session_mask
from strategy.ema import compute_ema
from strategy.ifvg import compute_ifvg
from strategy.cisd import compute_cisd
from strategy.signals import double_confirmation
from backtest.trade import Trade

PT_VALUE = 20.0; SWING = 8; RR = 1.5; MAX_TRADES_PER_DAY = 1  # PT_VALUE=$/pt; others from Pine

def backtest(df: pd.DataFrame, fill_mode: str = "next_open") -> list[Trade]:
    in_sess = in_session_mask(df.index)
    ifvg = compute_ifvg(df, in_sess)
    cisd = compute_cisd(df)
    ema = compute_ema(df, 20)
    sig = double_confirmation(ifvg, cisd)   # positional; align below
    o,h,l,c = (df[x].to_numpy(float) for x in ("open","high","low","close"))
    ema_v = ema.to_numpy(float); sess = in_sess.to_numpy(bool); sg = sig.to_numpy(object) if hasattr(sig,'to_numpy') else np.array(list(sig))
    idx = df.index; days = idx.tz_convert("America/New_York").date
    trades: list[Trade] = []; open_t = None; pending = None
    trades_today = 0; cur_day = None
    for i in range(SWING, len(df)):
        if days[i] != cur_day:
            cur_day = days[i]; trades_today = 0
        # 1) manage open trade on bar i (stop-first)
        if open_t is not None:
            _try_exit(open_t, h[i], l[i], idx[i], trades); 
            if open_t.outcome != "Open": open_t = None
        # 2) fill a pending entry at open[i], then check same-bar exit
        if open_t is None and pending is not None:
            open_t = _fill(pending, o[i] if fill_mode=="next_open" else pending["signal_close"], idx[i])
            pending = None; trades_today += 1
            _try_exit(open_t, h[i], l[i], idx[i], trades)
            if open_t.outcome != "Open": open_t = None
        # 3) evaluate a new signal at bar i (fills next bar).
        #    No NaN-EMA guard: ewm(adjust=False) seeds from bar 0 (no NaN); day-1 session is
        #    hundreds of bars in, so EMA is always warm. Signal only fires when ifvg state is a
        #    LIVE Bullish/Bearish (never "Expired"), which realizes Pine's 10-bar na(ifvgBottom/Top)
        #    entry-expiry gate (lines 513/523) — see the note in Step 4.
        if open_t is None and pending is None and sess[i] and trades_today < MAX_TRADES_PER_DAY:
            s = sg[i]
            if s == "Long" and c[i] > ema_v[i]:
                stop = float(np.min(l[i-SWING+1:i+1])); risk = c[i]-stop
                if risk > 0: pending = _mk("Long", c[i], stop, c[i]+risk*RR)
            elif s == "Short" and c[i] < ema_v[i]:
                stop = float(np.max(h[i-SWING+1:i+1])); risk = stop-c[i]
                if risk > 0: pending = _mk("Short", c[i], stop, c[i]-risk*RR)
    return trades
```
  Plus helpers: `_mk(direction, signal_close, stop, target)` returns a dict carrying `signal_close, stop, target, direction, risk`; `_fill(pending, price, t)` builds a `Trade(entry=price, ...)`. `_try_exit(trade, o, h, l, t, trades, counters)` applies **stop-first** logic (long: if `l<=stop` → Loss; elif `h>=target` → Win; short mirror), with **gap-through fills**: a stop that gaps through fills at the **worse** of stop/open (long: `exit=min(stop,o)`; short: `exit=max(stop,o)`); a target fills exactly at `target` (limit, no gap improvement — pessimistic/honest). Sets `exit/exit_time/outcome`, `pnl_usd=(exit-entry)*(+1 long/-1 short)*PT_VALUE`, `r_multiple=pnl_usd/(trade.risk*PT_VALUE)`, appends to `trades`. **Count** (in `counters`) how many exits are **same-bar-span** (bar hit both stop and target — resolved stop-first) so the runner can report the size of the pessimistic bias. Do **not** check exit on the signal bar itself (fill is the next bar). Swing windows are inclusive of the signal bar (`l[i-SWING+1:i+1]`). **`risk>0` guard:** skip only the degenerate `risk==0` case (8-bar swing == signal close); `risk<0` is impossible (swing includes the current bar). This is a deliberate, negligible divergence from Pine (which would submit a zero-risk entry).

- [ ] **Step 5: Run `pytest tests/test_engine.py -q`.** Expected: PASS. Iterate synthetic fixtures until the intended trades realize; keep assertions.

- [ ] **Step 6: Run `pytest tests/ -q`.** Expected: all green. **Commit:** `feat: faithful bar-by-bar backtest engine`.

---

## Task 4: Validate generated trades vs the real logs

**Files:** Create `validate_trades.py`; Test `tests/test_validate_trades.py`.

**Interfaces:**
- Produces: `parse_tv_log(path) -> pd.DataFrame` (columns `entry_date, direction, entry, exit, pnl_usd`); `compare(generated: list[Trade], real: pd.DataFrame, win_start, win_end) -> dict`. **Both** the generated and real sets are filtered to `[win_start, win_end]` before comparison (the runner passes, per log, `win_start = max(log_date_min, "2022-12-26")` and `win_end = min(log_date_max, "2025-12-11")` — so the two **disjoint** logs never see each other's generated trades). Keys: `win_start, win_end, n_real_in_window, n_real_excluded, n_generated_in_window, n_matched, n_missed, n_extra, precision, recall, matched_entry_price_delta, aggregate` where `aggregate` = per-side PF/WR/total_pnl/direction_mix for **generated vs real, both on the same window**. `matched_entry_price_delta` is a **secondary diagnostic only** (see the data-adjustment caveat in Global Constraints) — never a pass/fail gate.

- [ ] **Step 1: Write `tests/test_validate_trades.py` (failing).** (a) `parse_tv_log` pairs Entry/Exit rows into one trade, derives `direction` from the Entry `Type`, and the PnL identity `pnl_usd == (exit-entry)*dir*20` holds (2-trade synthetic CSV). (b) **Window-match + disjointness:** real set = 3 trades (one after `win_end`, one before `win_start`); generated set = a trade INSIDE the window matching a real date+dir, a generated trade OUTSIDE the window (must NOT count as extra), and an in-window generated extra → assert `n_real_in_window==1, n_real_excluded==2, n_generated_in_window==2, n_matched==1, n_missed==0, n_extra==1, precision==0.5`. (c) opposite-direction on a matched date scores as a **mismatch** (missed + extra), not a match.

- [ ] **Step 2: Run it.** Expected: FAIL.

- [ ] **Step 3: Implement `validate_trades.py`.** `parse_tv_log`: read CSV; Entry/Exit rows pair on `Trade #`; `direction = "Long" if "long" in Type.lower() else "Short"`; `entry_date` = NY date of the Entry `Date and time`; `entry`/`exit` from the paired `Price USD`; `pnl_usd` from `Net P&L USD`. `compare`: filter **both** sets to `win_start <= entry_date <= win_end`; `n_real_excluded` = real trades outside `[win_start, win_end]`; build `(entry_date, direction)` keys; matched = keys in both, missed = real-only, extra = generated-only; `precision = matched/(matched+extra)`, `recall = matched/(matched+missed)`; for matched pairs collect `generated.entry - real.entry` (diagnostic); compute per-side aggregates (PF = gross win/|gross loss|, WR, total_pnl, direction mix) via reused `mc.metrics`. Return the dict. **Do not** filter or tune generated trades to improve the match — only window-clip.

- [ ] **Step 4: Run `pytest tests/test_validate_trades.py -q`.** Expected: PASS. **Commit:** `feat: validate generated trades vs real logs`.

---

## Task 5: Runner + results.json + charts

**Files:** Create `run_backtest.py`; output `backtest_results.json` (committed), `charts/` PNGs (committed). Test: extend a smoke check.

- [ ] **Step 1: Write `run_backtest.py`.** `load_nq()` → `backtest(df)` (capture the same-bar-span exit count) → parse both real logs → for **each** log compute its window `win_start = max(log_date_min, "2022-12-26")`, `win_end = min(log_date_max, "2025-12-11")` and call `compare(generated, real, win_start, win_end)`. Assemble a results dict: total generated trade count, generated aggregate (PF/WR/total_pnl/max_drawdown via `mc.metrics`), the same-bar-span count, and the two per-log `compare` reports. **Sanity-assert the known in-window real baselines** (2023-24 → 95 trades/−$4,600; winning → 59 trades/+$18,115) so a parsing/window regression fails loudly. Write `backtest_results.json`. Charts (matplotlib): (1) generated equity curve; (2) coverage bars (real-in-window / matched / missed / extra) per log; (3) generated-vs-real aggregate PF & WR per log, on the shared window. Save to `charts/`.
- [ ] **Step 2: Run `python run_backtest.py`.** Requires the Phase-1 raw data present (`data/raw/…`). Expected: writes `backtest_results.json` + PNGs, no exceptions. Capture the real coverage numbers.
- [ ] **Step 3: Add `tests/test_smoke_run.py`** — run `backtest` on a ~2-day real slice (via `load_nq` then `.loc[...]`) and assert it returns a `list[Trade]` without error; skip if raw data absent.
- [ ] **Step 4: Run `pytest tests/ -q`.** Expected: green. **Commit:** `feat: backtest runner + results.json + charts`.

---

## Task 6: Notebook + writeup + final review/merge

**Files:** Create `notebooks/03_strategy_engine.ipynb`, `WRITEUP_STRATEGY.md`; update `README.md`.

- [ ] **Step 1: `WRITEUP_STRATEGY.md`** — plain-English: what was rebuilt, the reused ports vs newly-built transition logic (and the corrected CISD off-by-one), and the honest faithfulness result (coverage %, generated-vs-real PF/WR per log **on the shared in-window baselines: 95/−$4,600 and 59/+$18,115** — explicitly note the 13 winning-log trades / +$10,285 excluded past the Dec-2025 data edge). Caveats, stated plainly: **(a) the data is a back-adjusted continuous NQ series, offset from the unadjusted logs, so absolute prices/$-PnL are NOT directly comparable — headline metrics are trade-date+direction coverage and win/loss outcomes, which the offset does not affect; the entry-price delta is a data-adjustment artifact, NOT fill granularity;** (b) 1-minute bars vs the logs' tick fills; (c) the ports are unvalidated against TradingView (no premium — the program's founding constraint), so fidelity rests on Pine-logic unit tests + this real-log validation; (d) `compute_ema` runs over the continuous ETH series, which only equals Pine `ta.ema` if the reference chart was continuous. Lead with the honest finding, not a flattering one.
  - **Optional data-reconciliation diagnostic:** compute the per-day offset between load_nq and each log on matched trade-days; report whether it is ~additive (point-PnL invariant) or ~multiplicative (flag the ~4–21% $-PnL scaling). This substantiates caveat (a) rather than asserting it.
- [ ] **Step 2: `notebooks/03_strategy_engine.ipynb`** — self-contained: load data, run backtest, show the coverage tables + charts, narrate the validation. Restart-and-run-all clean.
- [ ] **Step 3: Update `README.md`** — add the Phase-2 section (what it does, how to run, headline coverage numbers).
- [ ] **Step 4: Final whole-branch review** — dispatch superpowers:requesting-code-review over the branch diff; fix Critical/Important findings.
- [ ] **Step 5:** `pytest tests/ -q` green → **finish branch** (merge `feat/phase2-strategy-engine` → `master`) → update the vault (`15-fyp-strategy-engine/_INDEX.md` Phase-2 done + honest result; `14-monte-carlo` cross-link if relevant).

---

## Self-review notes
- **Spec coverage:** session gate (T1), transition trigger (T2), execution/stop/target/1-per-day/flat-only/no-lookahead (T3), validation + coverage accounting (T4), results/charts (T5), writeup/notebook/merge (T6) — all mapped.
- **Fidelity risks flagged for the final review:** the corrected **CISD neighbor index** (Blocker 4), the `ifvg` session gate, the `double_confirmation` truth table, the 8-bar **inclusive** swing window, gap-through exit fills, and **stop-first** same-bar resolution.
- **Honesty (post-adversarial-review):** (1) data is a back-adjusted continuation, so headline = **coverage + outcomes**, not raw prices/$-PnL (Blocker 1); (2) `fill_mode` fixed a-priori to `next_open`, never selected by log agreement (Blocker 2); (3) generated set window-matched to each **disjoint** log; baselines are the **in-window** 95/−$4,600 and 59/+$18,115, not the full logs (Blocker 3); (4) both logs scored (incl. the losing one); no hard pass/fail % baked in.
- **This plan was revised after a 3-lens adversarial review (Pine fidelity / validation rigor / engine correctness) that returned NO-GO-as-written; all 4 blockers were verified against the real data and fixed above before build.**
