"""Bar-by-bar backtest execution loop for the FYP IFVG+CISD NQ strategy.

From docs/superpowers/plans/2026-07-13-phase2-strategy-engine.md, Task 3
("Backtest engine"). Reuses the strategy ports (session/ema/ifvg/cisd) and
the double-confirmation transition trigger (Task 1-2) to drive a faithful
per-bar simulation: session-gated entries, EMA-filtered direction, an
8-bar swing stop (inclusive of the signal bar), a 1.5R target, 1 trade/day,
1 contract, and a stop-first / gap-through exit fill model.

Three phases run per bar i, strictly in this order:
  1. Manage the currently open trade (stop-first exit check on bar i).
  2. Fill a pending entry at open[i] (next_open mode), then check for a
     same-bar exit on that same bar.
  3. Evaluate a NEW signal at bar i (it can only fill on a LATER bar) --
     the signal bar itself is never exit-checked.

No lookahead: every decision at bar i only reads bars <= i.

Phase 4 (docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md, Task 1
Step 5) split this into a cacheable signal layer and a cheap execution
layer, and threaded a `StrategyParams` through both so the engine can be
re-run cheaply over many (params, window) combinations for walk-forward
tuning. `backtest(df, params=StrategyParams())` = the Phase-2 default
behavior, byte-identical for defaults.
"""

import numpy as np
import pandas as pd

from strategy.session import in_session_mask
from strategy.ema import compute_ema
from strategy.ifvg import compute_ifvg
from strategy.cisd import compute_cisd
from strategy.signals import double_confirmation
from strategy.params import StrategyParams
from backtest.trade import Trade

PT_VALUE = 20.0            # USD per NQ index point
MAX_TRADES_PER_DAY = 1     # Pine `maxTradesPerDay`


def compute_signal_layer(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> dict:
    """Precompute the (cacheable) signal layer: session mask, IFVG/CISD
    double-confirmation signal, EMA, and the raw OHLC/day/index arrays that
    `run_execution` needs. Pure function of `df` and `params` -- no
    execution-loop state -- so it can be computed once and sliced/reused
    across many execution runs (e.g. walk-forward IS/OOS windows).
    """
    in_sess = in_session_mask(df.index, params.session_start, params.session_end)
    ifvg = compute_ifvg(df, in_sess, params.fvg_threshold)
    cisd = compute_cisd(df)
    ema = compute_ema(df, params.ema_length)
    sig = double_confirmation(ifvg, cisd)  # positional; aligned by row position, not index

    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    ema_v = ema.to_numpy(dtype=float)
    sess = in_sess.to_numpy(dtype=bool)
    sg = sig.to_numpy()

    idx = df.index
    days = idx.tz_convert("America/New_York").date

    return {
        "sig": sg,
        "ema_v": ema_v,
        "sess": sess,
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "days": days,
        "index": idx,
    }


def run_execution(layer: dict, params: StrategyParams = StrategyParams(), fill_mode: str = "next_open") -> list[Trade]:
    """Run the bar-by-bar simulation over a precomputed signal layer and
    return the list of CLOSED trades.

    Trades still open at the end of the data are intentionally not
    appended (see Task 3 spec: only resolved trades are returned).

    Exposes a same-bar-span diagnostic as a function attribute --
    `run_execution.same_bar_span_count` -- set at the end of each call to
    the number of exits where a single bar's range touched both the stop
    and the target (resolved stop-first).
    """
    sg = layer["sig"]
    ema_v = layer["ema_v"]
    sess = layer["sess"]
    o, h, l, c = layer["o"], layer["h"], layer["l"], layer["c"]
    days = layer["days"]
    idx = layer["index"]

    swing = params.swing_lookback
    rr = params.rr
    n = len(sg)

    trades: list[Trade] = []
    counters = {"same_bar_span": 0}
    open_t: Trade | None = None
    pending: dict | None = None
    trades_today = 0
    cur_day = None

    for i in range(swing, n):
        if days[i] != cur_day:
            cur_day = days[i]
            trades_today = 0

        # 1) manage the open trade on bar i (stop-first, gap-through)
        if open_t is not None:
            _try_exit(open_t, o[i], h[i], l[i], idx[i], trades, counters)
            if open_t.outcome != "Open":
                open_t = None

        # 2) fill a pending entry at open[i] (next_open mode), then check
        #    for a same-bar exit
        if open_t is None and pending is not None:
            fill_price = o[i] if fill_mode == "next_open" else pending["signal_close"]
            open_t = _fill(pending, fill_price, idx[i])
            pending = None
            trades_today += 1
            _try_exit(open_t, o[i], h[i], l[i], idx[i], trades, counters)
            if open_t.outcome != "Open":
                open_t = None

        # 3) evaluate a NEW signal at bar i (fills next bar; the signal bar
        #    itself is never exit-checked). No NaN-EMA guard: ewm(adjust=False)
        #    seeds from bar 0, so EMA is always warm.
        if open_t is None and pending is None and sess[i] and trades_today < MAX_TRADES_PER_DAY:
            s = sg[i]
            if s == "Long" and c[i] > ema_v[i]:
                stop = float(np.min(l[i - swing + 1: i + 1]))
                risk = c[i] - stop
                if risk > 0:
                    pending = _mk("Long", c[i], stop, c[i] + risk * rr)
            elif s == "Short" and c[i] < ema_v[i]:
                stop = float(np.max(h[i - swing + 1: i + 1]))
                risk = stop - c[i]
                if risk > 0:
                    pending = _mk("Short", c[i], stop, c[i] - risk * rr)

    run_execution.same_bar_span_count = counters["same_bar_span"]
    return trades


run_execution.same_bar_span_count = 0  # populated by each call; see run_execution()'s docstring


def backtest(df: pd.DataFrame, params: StrategyParams = StrategyParams(), fill_mode: str = "next_open") -> list[Trade]:
    """Run the bar-by-bar simulation and return the list of CLOSED trades.

    `backtest(df, params=StrategyParams())` = `run_execution(compute_signal_layer(df, params), params, fill_mode)`
    -- byte-identical to the pre-Phase-4 engine for default params.

    Exposes the same `backtest.same_bar_span_count` diagnostic as before
    (mirrored from `run_execution.same_bar_span_count` after each call), so
    existing callers don't need to change.
    """
    layer = compute_signal_layer(df, params)
    trades = run_execution(layer, params, fill_mode)
    backtest.same_bar_span_count = run_execution.same_bar_span_count
    return trades


backtest.same_bar_span_count = 0  # populated by each call; see backtest()'s docstring


def _mk(direction: str, signal_close: float, stop: float, target: float) -> dict:
    """Build a pending-entry record. `risk` is fixed at signal time (distance
    from the signal-bar close to the stop) and carried unchanged into the
    filled Trade, independent of the actual next-bar fill price."""
    return {
        "direction": direction,
        "signal_close": signal_close,
        "stop": stop,
        "target": target,
        "risk": abs(signal_close - stop),
    }


def _fill(pending: dict, price: float, t) -> Trade:
    return Trade(
        entry_time=t,
        direction=pending["direction"],
        entry=price,
        stop=pending["stop"],
        target=pending["target"],
        risk=pending["risk"],
    )


def _try_exit(trade: Trade, o: float, h: float, l: float, t, trades: list[Trade], counters: dict) -> None:
    """Stop-first exit check with gap-through fills.

    A stop that gaps through fills at the WORSE of stop/open (pessimistic,
    no slippage beyond the gap itself): long exit = min(stop, o); short
    exit = max(stop, o). A target fills exactly at the target price (limit
    fill, no gap improvement). If a single bar's range touches both stop
    and target, the stop wins (stop-first tie-break) and the event is
    counted into counters["same_bar_span"]. No-op (leaves the trade open)
    if neither is hit.
    """
    if trade.direction == "Long":
        stop_hit = l <= trade.stop
        target_hit = h >= trade.target
        if stop_hit and target_hit:
            counters["same_bar_span"] += 1
        if stop_hit:
            exit_price, outcome = min(trade.stop, o), "Loss"
        elif target_hit:
            exit_price, outcome = trade.target, "Win"
        else:
            return
        sign = 1.0
    else:  # Short
        stop_hit = h >= trade.stop
        target_hit = l <= trade.target
        if stop_hit and target_hit:
            counters["same_bar_span"] += 1
        if stop_hit:
            exit_price, outcome = max(trade.stop, o), "Loss"
        elif target_hit:
            exit_price, outcome = trade.target, "Win"
        else:
            return
        sign = -1.0

    trade.exit = exit_price
    trade.exit_time = t
    trade.outcome = outcome
    trade.pnl_usd = (exit_price - trade.entry) * sign * PT_VALUE
    trade.r_multiple = trade.pnl_usd / (trade.risk * PT_VALUE)
    trades.append(trade)
