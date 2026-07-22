"""Tests for backtest.costs.CostModel.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md) -- Blocker 1:
reason-aware costs, no differential under-costing of the new treatment arms.

Default constants: commission_rt=$5.00, slippage_ticks_entry=1,
slippage_ticks_stop=1, tick_value=$5.00/tick, multiplier=1.0 -- so each
"tick" of slippage costs $5, same as the commission.
"""

import pytest

from backtest.costs import LIMIT_EXIT_REASONS, MARKET_EXIT_REASONS, CostModel


def test_market_and_limit_reason_sets():
    assert MARKET_EXIT_REASONS == {"stop", "trail", "time", "partial_remainder_stop"}
    assert LIMIT_EXIT_REASONS == {"target", "partial_scaleout", "partial_remainder_target"}
    assert MARKET_EXIT_REASONS.isdisjoint(LIMIT_EXIT_REASONS)


def test_target_winner_pays_commission_plus_one_tick():
    cm = CostModel()
    # limit fill (target): 0 exit slippage -> commission($5) + entry tick($5) = $10
    assert cm.net_pnl(150.0, "target") == pytest.approx(150.0 - 10.0)


@pytest.mark.parametrize("reason", ["stop", "trail", "time"])
def test_market_exits_pay_commission_plus_two_ticks(reason):
    cm = CostModel()
    # market fill: commission($5) + entry tick($5) + exit tick($5) = $15 --
    # trail/time are NOT under-costed as if they were limit fills.
    assert cm.net_pnl(-100.0, reason) == pytest.approx(-100.0 - 15.0)


def test_partial_remainder_stop_is_a_market_exit_like_stop_trail_time():
    cm = CostModel()
    assert cm.net_pnl(-40.0, "partial_remainder_stop") == pytest.approx(-40.0 - 15.0)


def test_multiplier_zero_net_equals_gross():
    cm = CostModel(multiplier=0.0)
    assert cm.net_pnl(150.0, "target") == 150.0
    assert cm.net_pnl(-100.0, "stop") == -100.0


def test_multiplier_two_doubles_total_cost():
    cm1 = CostModel(multiplier=1.0)
    cm2 = CostModel(multiplier=2.0)
    gross = 200.0
    cost_1x = gross - cm1.net_pnl(gross, "stop")
    cost_2x = gross - cm2.net_pnl(gross, "stop")
    assert cost_2x == pytest.approx(cost_1x * 2.0)
    assert cost_1x == pytest.approx(15.0)
    assert cost_2x == pytest.approx(30.0)


def test_unknown_exit_reason_raises():
    cm = CostModel()
    with pytest.raises(ValueError):
        cm.net_pnl(100.0, "not_a_real_reason")


def test_partial_pays_two_commissions_with_correct_per_leg_slippage():
    """A partial_1R trade: leg 1 = the +1R scale-out (limit, gets the
    one-time entry slippage), leg 2 = the remainder (charge_entry=False --
    there is only one entry fill for the whole trade even though there are
    two exit fills). Net is computed per fill and summed (Global
    Constraints)."""
    cm = CostModel()
    leg1_gross, leg2_gross = 50.0, 30.0

    # remainder closes via its own stop (market) -----------------------
    leg1_net = cm.net_pnl(leg1_gross, "partial_scaleout", charge_entry=True)
    leg2_net = cm.net_pnl(leg2_gross, "partial_remainder_stop", charge_entry=False)
    # leg1: commission($5) + entry tick($5) + 0 (limit) = $10 -> net = $40
    # leg2: commission($5) + 0 (no entry) + exit tick($5) = $10 -> net = $20
    assert leg1_net == pytest.approx(40.0)
    assert leg2_net == pytest.approx(20.0)
    total_cost = (leg1_gross + leg2_gross) - (leg1_net + leg2_net)
    assert total_cost == pytest.approx(20.0)  # 2 commissions ($10) + 1 entry tick($5) + 1 exit tick($5)

    # remainder closes via its own target (limit) -----------------------
    leg2_net_target = cm.net_pnl(leg2_gross, "partial_remainder_target", charge_entry=False)
    # leg2: commission($5) + 0 + 0 (limit) = $5 -> net = $25
    assert leg2_net_target == pytest.approx(25.0)
