"""Tests for validate_trades.py -- generated-vs-real-log validation.

From the Phase-2 strategy-engine spec
(docs/specs/2026-07-12-phase2-strategy-engine-design.md):
(a) parse_tv_log pairs Entry/Exit rows and the PnL identity holds;
(b) window-match + disjointness, with exact asserted counts;
(c) opposite-direction on a matched date scores as a mismatch, not a match.
"""
from datetime import date, datetime

import pandas as pd
import pytest

from backtest.trade import Trade
from validate_trades import compare, parse_tv_log


def _trade(entry_time, direction, entry, pnl_usd):
    """Minimal generated Trade for compare() tests -- stop/target/risk are
    irrelevant to window-matching so they're filled with placeholders."""
    return Trade(
        entry_time=entry_time,
        direction=direction,
        entry=entry,
        stop=0.0,
        target=0.0,
        risk=1.0,
        exit_time=entry_time,
        exit=entry,
        pnl_usd=pnl_usd,
        r_multiple=0.0,
        outcome="Win" if pnl_usd >= 0 else "Loss",
    )


# --- (a) parse_tv_log pairs Entry/Exit rows; PnL identity holds -------------


def test_parse_tv_log_pairs_rows_and_pnl_identity(tmp_path):
    csv_path = tmp_path / "tv_log.csv"
    csv_path.write_text(
        "Trade #,Type,Date and time,Price USD,Net P&L USD\n"
        "1,Exit long,2024-01-05 10:15,105.0,100.0\n"
        "1,Entry long,2024-01-05 09:45,100.0,100.0\n"
        "2,Exit short,2024-01-06 10:00,48.0,40.0\n"
        "2,Entry short,2024-01-06 09:50,50.0,40.0\n"
    )

    df = parse_tv_log(str(csv_path))

    assert list(df.columns) == ["entry_date", "direction", "entry", "exit", "pnl_usd"]
    assert len(df) == 2
    assert set(df["direction"]) == {"Long", "Short"}
    assert df.loc[df["direction"] == "Long", "entry_date"].iloc[0] == date(2024, 1, 5)

    dir_sign = df["direction"].map({"Long": 1, "Short": -1})
    expected_pnl = (df["exit"] - df["entry"]) * dir_sign * 20
    assert (expected_pnl == df["pnl_usd"]).all()


# --- (b) window-match + disjointness -----------------------------------------


def test_window_match_excludes_out_of_window_and_scores_precision():
    win_start, win_end = date(2023, 2, 1), date(2023, 11, 30)

    real = pd.DataFrame([
        # inside the window -- will be matched by a generated trade below
        {"entry_date": date(2023, 6, 1), "direction": "Long", "entry": 100.0,
         "exit": 110.0, "pnl_usd": 200.0},
        # before win_start -- excluded, never scored as a miss
        {"entry_date": date(2023, 1, 1), "direction": "Short", "entry": 50.0,
         "exit": 48.0, "pnl_usd": 40.0},
        # after win_end -- excluded, never scored as a miss
        {"entry_date": date(2023, 12, 31), "direction": "Long", "entry": 200.0,
         "exit": 210.0, "pnl_usd": 200.0},
    ], columns=["entry_date", "direction", "entry", "exit", "pnl_usd"])

    generated = [
        # inside the window, matches the one in-window real trade
        _trade(datetime(2023, 6, 1, 9, 45), "Long", 101.0, 190.0),
        # outside the window -- must NOT count as an extra
        _trade(datetime(2023, 1, 15, 9, 45), "Short", 55.0, -20.0),
        # inside the window, no matching real trade -> extra
        _trade(datetime(2023, 7, 4, 9, 45), "Short", 60.0, -10.0),
    ]

    result = compare(generated, real, win_start, win_end)

    assert result["n_real_in_window"] == 1
    assert result["n_real_excluded"] == 2
    assert result["n_generated_in_window"] == 2
    assert result["n_matched"] == 1
    assert result["n_missed"] == 0
    assert result["n_extra"] == 1
    assert result["precision"] == 0.5


# --- (c) opposite direction on a matched date is a mismatch -----------------


def test_opposite_direction_same_date_is_mismatch_not_match():
    win_start, win_end = date(2023, 1, 1), date(2023, 12, 31)

    real = pd.DataFrame([
        {"entry_date": date(2023, 6, 1), "direction": "Long", "entry": 100.0,
         "exit": 110.0, "pnl_usd": 200.0},
    ], columns=["entry_date", "direction", "entry", "exit", "pnl_usd"])

    generated = [
        _trade(datetime(2023, 6, 1, 9, 45), "Short", 100.0, -150.0),
    ]

    result = compare(generated, real, win_start, win_end)

    assert result["n_matched"] == 0
    assert result["n_missed"] == 1
    assert result["n_extra"] == 1
