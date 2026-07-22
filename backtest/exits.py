"""Pluggable exit-mode handlers for the Phase-5 backtest engine.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md). Implements
the 4 NEW exit modes -- `breakeven_1R`, `trail_swing`,
`partial_1R`, `time_stop`. `fixed_1_5R` (the base/default mode) is NOT
reimplemented here: `backtest.engine.run_execution` routes it straight to
the original, untouched `backtest.engine._try_exit` so the Phase-2/4 base
path is provably byte-identical (same code, not a re-derivation of the same
behavior) -- see engine.py's dispatch and tests/test_engine_p5_regression.py.

State machine
-------------
Each handler manages ONE open trade across the bars it stays open, called
once per bar by `run_execution` via `manage_bar`. State lives in a plain
per-trade `dict` (`init_state`), not on `Trade` itself -- `Trade`'s schema
stays additive (`net_pnl`, `exit_reason` only; see
backtest/trade.py). Fields: `stop` (the CURRENT live stop level -- starts
at the initial stop; becomes breakeven/trail/remainder-breakeven once
activated), `activated_1r` (+1R reached), `half_closed` (partial_1R only:
the scale-out leg has fired), `remainder_target` (partial_1R only, the 3R
level), `banked_pnl_usd` (partial_1R only, the realized scale-out leg's
gross USD P&L, banked for the final blended P&L).

Signal-close anchoring
-----------------------
R and every managed level (1R, 1.5R, 3R) are anchored at the SIGNAL-BAR
CLOSE, exactly like the base engine's fixed target.
`Trade` doesn't store `signal_close` directly, but it's exactly
reconstructible from the two fields it DOES store: `risk = abs(signal_close
- stop)` was fixed at signal time (see engine.py's `_mk`), so
`signal_close = stop + risk` (Long) / `stop - risk` (Short) -- `_level`
below reconstructs it this way rather than adding a field.

Intrabar stop-first sequencing (Blocker 3)
-------------------------------------------
On every managed bar, each handler evaluates the CURRENT stop level FIRST,
using the same stop-first / gap-through-fills-at-the-worse-of-stop/open
rule as the base engine's `_try_exit`. If the bar breaches the current
stop, it closes as a stop-type exit immediately -- +1R activation, a
breakeven move, a trail ratchet, or a partial scale-out may NEVER also
happen on that same bar. Only a bar that does NOT breach the current stop
may activate/ratchet/scale-out. This is enforced structurally below: every
handler's stop-breach branch `return`s before any activation logic runs.
"""

from __future__ import annotations

import numpy as np

from backtest.trade import Trade

TIME_STOP_ET = "11:00"
_TSH, _TSM = (int(x) for x in TIME_STOP_ET.split(":"))
_TIME_STOP_MINUTES = _TSH * 60 + _TSM  # derived from TIME_STOP_ET, not a separate hardcode


def init_state(trade: Trade) -> dict:
    """Per-trade managed-exit state, created once at fill time (before the
    first `manage_bar` call for this trade)."""
    return {
        "stop": trade.stop,
        "activated_1r": False,
        "half_closed": False,
        "remainder_target": None,
        "banked_pnl_usd": 0.0,
    }


def manage_bar(
    trade: Trade,
    state: dict,
    exit_mode: str,
    i: int,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    idx,
    swing: int,
    pt_value: float,
    cost_model,
    trades: list[Trade],
    counters: dict,
) -> None:
    """Manage `trade` on bar `i` under `exit_mode`. Mutates `trade` in
    place and appends it to `trades` if it closes on this bar (leaves
    `trade.outcome == "Open"` and appends nothing otherwise); mutates
    `state` in place for activation/ratchet bookkeeping."""
    _HANDLERS[exit_mode](trade, state, i, o, h, l, c, idx, swing, pt_value, cost_model, trades, counters)


# --- shared helpers ----------------------------------------------------


def _sign(direction: str) -> float:
    return 1.0 if direction == "Long" else -1.0


def _stop_breach(direction: str, stop_level: float, h_i: float, l_i: float) -> bool:
    return l_i <= stop_level if direction == "Long" else h_i >= stop_level


def _favorable_hit(direction: str, h_i: float, l_i: float, level: float) -> bool:
    return h_i >= level if direction == "Long" else l_i <= level


def _gap_through_fill(direction: str, stop_level: float, o_i: float) -> float:
    """Worse of (stop, open) -- same gap-through convention as the base
    engine's `_try_exit`."""
    return min(stop_level, o_i) if direction == "Long" else max(stop_level, o_i)


def _level(trade: Trade, multiple: float) -> float:
    """Price `multiple` R away from the SIGNAL-BAR CLOSE (reconstructed
    from `trade.stop`/`trade.risk`; see module docstring), toward profit
    for positive `multiple`."""
    signal_close = trade.stop + trade.risk if trade.direction == "Long" else trade.stop - trade.risk
    r = trade.risk
    return signal_close + multiple * r if trade.direction == "Long" else signal_close - multiple * r


def _close(trade: Trade, price: float, t, reason: str, pt_value: float, cost_model, trades: list[Trade]) -> None:
    """Finalize a single-fill exit: set exit/exit_time/outcome/pnl_usd/
    r_multiple/exit_reason/net_pnl and append to `trades`. `outcome` is
    Win/Loss by the SIGN of the realized P&L (not by exit type) so a
    profitable trailing-stop or breakeven exit isn't mislabeled -- unlike
    the base engine's `_try_exit`, which is untouched and keeps its own
    "stop always Loss / target always Win" convention for the regression-
    locked fixed_1_5R path."""
    sign = _sign(trade.direction)
    pnl = (price - trade.entry) * sign * pt_value
    trade.exit = price
    trade.exit_time = t
    trade.pnl_usd = pnl
    trade.outcome = "Win" if pnl > 0 else "Loss"
    trade.r_multiple = pnl / (trade.risk * pt_value)
    trade.exit_reason = reason
    trade.net_pnl = pnl if cost_model is None else cost_model.net_pnl(pnl, reason)
    trades.append(trade)


def _close_partial(
    trade: Trade, remainder_price: float, t, remainder_reason: str,
    pt_value: float, cost_model, trades: list[Trade], state: dict,
) -> None:
    """Finalize partial_1R's two-leg exit: blend the already-banked
    scale-out leg with the remainder leg's gross P&L, and (if `cost_model`)
    sum the two legs' NET P&L separately (`charge_entry=False` on the
    remainder leg -- there is only one entry fill for the whole trade)."""
    sign = _sign(trade.direction)
    remainder_gross = (remainder_price - trade.entry) * sign * pt_value * 0.5
    gross_total = state["banked_pnl_usd"] + remainder_gross
    trade.exit = remainder_price
    trade.exit_time = t
    trade.pnl_usd = gross_total
    trade.outcome = "Win" if gross_total > 0 else "Loss"
    trade.r_multiple = gross_total / (trade.risk * pt_value)
    trade.exit_reason = remainder_reason
    if cost_model is None:
        trade.net_pnl = gross_total
    else:
        leg1_net = cost_model.net_pnl(state["banked_pnl_usd"], "partial_scaleout", charge_entry=True)
        leg2_net = cost_model.net_pnl(remainder_gross, remainder_reason, charge_entry=False)
        trade.net_pnl = leg1_net + leg2_net
    trades.append(trade)


# --- per-mode handlers ---------------------------------------------------


def _manage_breakeven(trade, state, i, o, h, l, c, idx, swing, pt_value, cost_model, trades, counters):
    o_i, h_i, l_i, t_i = o[i], h[i], l[i], idx[i]
    stop_level = state["stop"]
    stop_hit = _stop_breach(trade.direction, stop_level, h_i, l_i)
    target_hit = _favorable_hit(trade.direction, h_i, l_i, trade.target)
    if stop_hit and target_hit:
        counters["same_bar_span"] += 1
    if stop_hit:
        # Blocker 3: this bar breached the CURRENT stop -- resolve as a
        # stop, never as a breakeven-triggered exit, even if the bar's
        # range also reaches +1R.
        price = _gap_through_fill(trade.direction, stop_level, o_i)
        _close(trade, price, t_i, "stop", pt_value, cost_model, trades)
        return
    if target_hit:
        _close(trade, trade.target, t_i, "target", pt_value, cost_model, trades)
        return
    if not state["activated_1r"]:
        r1 = _level(trade, 1.0)
        if _favorable_hit(trade.direction, h_i, l_i, r1):
            state["activated_1r"] = True
            state["stop"] = trade.entry  # breakeven = the trade's own fill price


def _manage_trail(trade, state, i, o, h, l, c, idx, swing, pt_value, cost_model, trades, counters):
    o_i, h_i, l_i, t_i = o[i], h[i], l[i], idx[i]
    stop_level = state["stop"]
    stop_hit = _stop_breach(trade.direction, stop_level, h_i, l_i)
    if stop_hit:
        # Blocker 3: no target check needed (trail_swing has no fixed
        # target) -- just resolve the stop. Pre-activation this is the
        # original swing stop ("stop"); post-activation it's the trailing
        # level ("trail") -- either way, no ratchet happens on this bar.
        price = _gap_through_fill(trade.direction, stop_level, o_i)
        reason = "trail" if state["activated_1r"] else "stop"
        _close(trade, price, t_i, reason, pt_value, cost_model, trades)
        return

    # Not stopped: this bar's causal swing level (identical window formula
    # to the initial stop -- bars [i-swing+1, i], never reaching before
    # index 0).
    window_lo = max(0, i - swing + 1)
    if trade.direction == "Long":
        swing_level = float(np.min(l[window_lo: i + 1]))
    else:
        swing_level = float(np.max(h[window_lo: i + 1]))

    if not state["activated_1r"]:
        r1 = _level(trade, 1.0)
        if _favorable_hit(trade.direction, h_i, l_i, r1):
            state["activated_1r"] = True
            # Seed the trail at the fresh swing level, but never worse than
            # the stop it replaces (ratchet only ever moves favorably).
            state["stop"] = (
                max(stop_level, swing_level) if trade.direction == "Long" else min(stop_level, swing_level)
            )
    else:
        # Already trailing: ratchet forward only -- a trailing stop that
        # could retreat isn't a stop.
        state["stop"] = (
            max(stop_level, swing_level) if trade.direction == "Long" else min(stop_level, swing_level)
        )


def _activate_partial(trade: Trade, state: dict, r1_price: float, pt_value: float) -> None:
    sign = _sign(trade.direction)
    state["banked_pnl_usd"] = (r1_price - trade.entry) * sign * pt_value * 0.5
    state["half_closed"] = True
    state["activated_1r"] = True
    state["stop"] = trade.entry  # remainder -> breakeven
    state["remainder_target"] = _level(trade, 3.0)


def _manage_partial(trade, state, i, o, h, l, c, idx, swing, pt_value, cost_model, trades, counters):
    o_i, h_i, l_i, t_i = o[i], h[i], l[i], idx[i]
    stop_level = state["stop"]
    stop_hit = _stop_breach(trade.direction, stop_level, h_i, l_i)

    if stop_hit:
        # Blocker 3: whichever phase we're in, the stop wins -- no scale-out
        # credit on the SAME bar that breached the (pre- or post-scale-out)
        # stop.
        price = _gap_through_fill(trade.direction, stop_level, o_i)
        if not state["half_closed"]:
            _close(trade, price, t_i, "stop", pt_value, cost_model, trades)  # full 1-unit loss, nothing banked yet
        else:
            _close_partial(trade, price, t_i, "partial_remainder_stop", pt_value, cost_model, trades, state)
        return

    if not state["half_closed"]:
        # partial_1R has no fixed 1.5R target for the whole unit -- the
        # only two things that can happen pre-activation are the stop
        # (handled above) or reaching +1R.
        r1 = _level(trade, 1.0)
        if _favorable_hit(trade.direction, h_i, l_i, r1):
            _activate_partial(trade, state, r1, pt_value)
        return

    # Remainder phase (0.5 unit): stop = breakeven (already in state["stop"]
    # and checked above), target = 3R.
    if _favorable_hit(trade.direction, h_i, l_i, state["remainder_target"]):
        _close_partial(trade, state["remainder_target"], t_i, "partial_remainder_target", pt_value, cost_model, trades, state)


def _et_minutes(ts) -> int:
    ny = ts.tz_convert("America/New_York") if ts.tzinfo is not None else ts
    return ny.hour * 60 + ny.minute


def _manage_time_stop(trade, state, i, o, h, l, c, idx, swing, pt_value, cost_model, trades, counters):
    o_i, h_i, l_i, c_i, t_i = o[i], h[i], l[i], c[i], idx[i]
    stop_level = state["stop"]  # fixed_1_5R's stop, never moves
    stop_hit = _stop_breach(trade.direction, stop_level, h_i, l_i)
    target_hit = _favorable_hit(trade.direction, h_i, l_i, trade.target)
    if stop_hit and target_hit:
        counters["same_bar_span"] += 1
    if stop_hit:
        price = _gap_through_fill(trade.direction, stop_level, o_i)
        _close(trade, price, t_i, "stop", pt_value, cost_model, trades)
        return
    if target_hit:
        _close(trade, trade.target, t_i, "target", pt_value, cost_model, trades)
        return
    if _et_minutes(t_i) >= _TIME_STOP_MINUTES:
        _close(trade, c_i, t_i, "time", pt_value, cost_model, trades)


_HANDLERS = {
    "breakeven_1R": _manage_breakeven,
    "trail_swing": _manage_trail,
    "partial_1R": _manage_partial,
    "time_stop": _manage_time_stop,
}
