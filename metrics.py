"""Pure scoring functions over a sequence of per-trade P&L (USD).

Vendored (unchanged logic) from
C:/Users/Alex/Projects/Trading-Strategy-Monte-Carlo-Simulation/mc/metrics.py
because that module lives in a separate, unrelated repo and importing across
repos would require sys.path surgery. Only the three functions
validate_trades.py needs are copied here: profit_factor, win_rate,
total_pnl. See the source file for the fuller metrics set (equity_curve,
max_drawdown, longest_losing_streak) if this repo ever needs those too.
"""
from __future__ import annotations
import math
from typing import Sequence


def total_pnl(pnls: Sequence[float]) -> float:
    return float(sum(pnls))


def win_rate(pnls: Sequence[float]) -> float:
    if len(pnls) == 0:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def profit_factor(pnls: Sequence[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
