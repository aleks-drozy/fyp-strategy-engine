"""Phase-6 Task 2: InstrumentSpec + engine threading.

The one thing that MUST hold: the NQ spec is the default everywhere, so all
Phase 2-5 behavior is byte-preserved (the golden-fixture regression in
test_engine_p5_regression.py and the 605-trade full-data check both continue
to run against the default path). These tests add the per-instrument cost
math and P&L threading on top.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import backtest, PT_VALUE
from strategy.instrument import InstrumentSpec, SPECS
from strategy.params import StrategyParams


def test_specs_frozen_values():
    assert SPECS["ES"] == InstrumentSpec("ES", 0.25, 12.50, 50.0)
    assert SPECS["NQ"] == InstrumentSpec("NQ", 0.25, 5.00, 20.0)
    assert SPECS["YM"] == InstrumentSpec("YM", 1.00, 5.00, 5.0)
    assert SPECS["NQ"].pt_value == PT_VALUE  # the engine default IS the NQ spec
    with pytest.raises(Exception):
        SPECS["ES"].pt_value = 1.0  # frozen


def test_per_instrument_cost_math():
    # a stop-exit loser pays commission + entry tick + exit tick, valued per instrument
    ym = SPECS["YM"].cost_model()
    es = SPECS["ES"].cost_model()
    nq = SPECS["NQ"].cost_model()
    assert ym.leg_cost("stop") == 5.0 + 5.0 + 5.0            # $5 comm + 2 ticks x $5
    assert es.leg_cost("stop") == 5.0 + 12.50 + 12.50        # $5 comm + 2 ticks x $12.50
    assert nq.leg_cost("stop") == 5.0 + 5.0 + 5.0            # unchanged Phase-5 NQ costs
    # a target (limit) exit pays no exit slippage
    assert es.leg_cost("target") == 5.0 + 12.50
    # multiplier scales
    assert SPECS["ES"].cost_model(multiplier=2.0).leg_cost("stop") == 2 * (5.0 + 25.0)
    assert SPECS["ES"].cost_model(multiplier=0.0).leg_cost("stop") == 0.0


def _one_trade_frame():
    """Synthetic session frame that yields exactly one long target-hit trade
    (mirrors the Phase-5 engine-test fixture approach: signals monkeypatched)."""
    idx = pd.date_range("2025-01-21 09:30", periods=40, freq="1min", tz="US/Eastern")
    base = np.full(len(idx), 100.0)
    df = pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5,
                       "close": base, "volume": 0}, index=idx)
    # bars 0-9 flat; signal fires at bar 9 (via monkeypatch); fill at bar 10 open;
    # bar 12 runs to the 1.5R target
    df.iloc[12, df.columns.get_loc("high")] = 120.0
    return df


def test_pt_value_threads_through_pnl(monkeypatch):
    import backtest.engine as eng

    def fake_signals(ifvg, cisd):
        n = len(ifvg)
        out = [""] * n
        out[9] = "Long"
        return pd.Series(out)

    monkeypatch.setattr(eng, "double_confirmation", fake_signals)
    monkeypatch.setattr(eng, "compute_ema", lambda df, period: pd.Series(0.0, index=df.index))

    df = _one_trade_frame()
    t_nq = backtest(df, StrategyParams(), spec=SPECS["NQ"])
    t_es = backtest(df, StrategyParams(), spec=SPECS["ES"])
    t_ym = backtest(df, StrategyParams(), spec=SPECS["YM"])
    assert len(t_nq) == len(t_es) == len(t_ym) == 1
    # identical trades in POINTS (entry/stop/target/exit)...
    for a, b in ((t_nq[0], t_es[0]), (t_nq[0], t_ym[0])):
        assert (a.entry, a.stop, a.target, a.exit) == (b.entry, b.stop, b.target, b.exit)
    # ...but P&L scaled by each instrument's $/pt (same point move)
    pts = (t_nq[0].exit - t_nq[0].entry)
    assert t_nq[0].pnl_usd == pytest.approx(pts * 20.0)
    assert t_es[0].pnl_usd == pytest.approx(pts * 50.0)
    assert t_ym[0].pnl_usd == pytest.approx(pts * 5.0)
    # r_multiple is pt_value-invariant (unit-risk) -- the Phase-6 gating statistic relies on this
    assert t_nq[0].r_multiple == pytest.approx(t_es[0].r_multiple) == pytest.approx(t_ym[0].r_multiple)


def test_pinned_phase5_value_reproduces_with_threaded_engine():
    """The strongest behavior-preservation lock: the exact Phase-5 F4 tuned
    net OOS PF (committed in phase5_results.json before the Phase-6 refactor)
    must reproduce through the full walk-forward stack with the spec-threaded
    engine at its NQ default. Slow (~90s) but load-bearing."""
    import os
    if not os.path.exists("data/raw/Dataset_NQ_1min_2022_2025.csv"):
        pytest.skip("Phase-1 raw data absent")
    from nqdata.load import load_nq
    from tuning.grid_p5 import build_grid_p5
    from tuning.walkforward_p5 import walk_forward_p5
    from tuning.walkforward import make_folds
    from backtest.costs import CostModel

    r = walk_forward_p5(load_nq(), build_grid_p5(), make_folds(), CostModel())
    pf = r["folds"][3]["oos_net_metrics"]["net_profit_factor"]
    assert pf == pytest.approx(2.2480132450331127, rel=1e-12)
