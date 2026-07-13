# fyp-strategy-engine

Rebuilding + extending the FYP IFVG+CISD NQ strategy in Python. Phase 1: data foundation.

## Phase 2 — Strategy engine + real-log validation

A faithful, bar-by-bar Python reimplementation of the FYP IFVG+CISD NQ
strategy (session gate, IFVG, CISD, EMA filter, double-confirmation entry,
8-bar swing stop, 1.5R target, 1 trade/day), run with the Pine script's
default parameters over the full Phase-1 dataset and validated against two
real TradingView trade logs.

**Honest headline:** the rebuild recovers 76-80% of the real logs'
trade-days and directions (good recall), but fires ~4x as many trades as
the real logs (precision only 20-25%) and underperforms them on profit
factor and win rate on both a losing and a winning period. It is
directionally consistent on both regimes, which supports the port being
substantively correct — the gap looks like tuning/selectivity in the real
"optimised" track record that the raw default parameters don't capture.
This motivates Phase 4 (parameter sweeps / regime filter). Full result,
interpretation, and data-comparability caveats: see `WRITEUP_STRATEGY.md`.

| | 2023-24 log (losing) | Winning log |
|---|---|---|
| Real baseline (in-window) | 95 trades / −$4,600 | 59 trades / +$18,115 |
| Matched / Missed / Extra | 76 / 19 / 300 | 45 / 14 / 134 |
| Precision / Recall | 0.20 / 0.80 | 0.25 / 0.76 |
| Generated PF / WR | 0.71 / 33.8% | 1.09 / 42.5% |
| Real PF / WR | 0.90 / 37.9% | 1.53 / 55.9% |

**Run it:**

```
.venv/Scripts/python run_backtest.py
```

Requires the Phase-1 raw data at `data/raw/Dataset_NQ_1min_2022_2025.csv`
(not committed) and the two real trade-log CSVs referenced in
`run_backtest.py`. Writes `backtest_results.json` and three charts to
`charts/` (equity curve, coverage bars, generated-vs-real PF/WR). Takes
~20 seconds — the engine is a pure bar-by-bar loop (no lookahead) over
~1.05M 1-minute rows.

See `notebooks/03_strategy_engine.ipynb` for the same run narrated
end-to-end with the coverage tables and charts inline, and
`WRITEUP_STRATEGY.md` for the full writeup.
