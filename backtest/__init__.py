"""Bar-by-bar backtest execution engine for the FYP IFVG+CISD NQ strategy.

Drives strategy.* (session gate, EMA filter, IFVG/CISD state machines, and
the double-confirmation transition trigger) through a sequential per-bar
loop that simulates fills, stops, and targets with no lookahead.
See backtest/engine.py for the algorithm and backtest/trade.py for the
Trade record.
"""
