"""Reason-aware trading cost model: commission + slippage applied to a
Trade's gross P&L.

From the Phase-5 exits/costs/vol-filter spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md): reason-aware
costs, with no differential under-costing of the new treatment arms.
Pre-registered constants (frozen; not tuned): TICK_VALUE=5.0 USD/tick,
COMMISSION_RT=5.0 USD, SLIPPAGE_TICKS_ENTRY=1, SLIPPAGE_TICKS_EXIT=1.

Slippage is charged on EVERY market-order fill, not only the literal "stop"
reason -- `MARKET_EXIT_REASONS` also includes "trail", "time" and
"partial_remainder_stop" (all adverse-side market fills), each paying 1
exit tick. True LIMIT fills -- "target", the partial's +1R scale-out
("partial_scaleout"), and the partial's 3R remainder target
("partial_remainder_target") -- pay 0 exit slippage. Treating trail/time
exits as if they were free limit fills would under-cost exactly the new
treatment arms this phase is testing, biasing the net comparison in their
favor.

Multi-leg trades (partial_1R): there is only ONE entry fill per trade even
though there are TWO exit fills (the +1R scale-out and the remainder), so
entry slippage is charged ONCE, not once per leg. Commission, by contrast,
is charged PER FILL (a broker commission per contract/exit event), so a
partial trade pays two commissions -- one per leg. The engine (backtest/
exits.py) calls `net_pnl` once per leg (`charge_entry=True` on the first,
`charge_entry=False` on the rest) and sums the results -- the spec charges
commission per fill, with per-leg slippage.
"""

from dataclasses import dataclass

MARKET_EXIT_REASONS = frozenset({"stop", "trail", "time", "partial_remainder_stop"})
LIMIT_EXIT_REASONS = frozenset({"target", "partial_scaleout", "partial_remainder_target"})


@dataclass(frozen=True)
class CostModel:
    commission_rt: float = 5.0
    slippage_ticks_entry: int = 1
    slippage_ticks_stop: int = 1  # per-market-exit-fill slippage, in ticks
    tick_value: float = 5.0
    multiplier: float = 1.0  # cost-sensitivity dial: 0.0/1.0/2.0 (reported, never used for selection)

    def leg_cost(self, exit_reason: str, charge_entry: bool = True) -> float:
        """Total USD cost (commission + slippage) for ONE fill leg with the
        given `exit_reason`, before rounding -- already scaled by
        `multiplier`. `charge_entry=False` skips the one-time entry
        slippage for a later leg of a multi-leg trade (see module
        docstring)."""
        if exit_reason not in MARKET_EXIT_REASONS and exit_reason not in LIMIT_EXIT_REASONS:
            raise ValueError(f"unknown exit_reason: {exit_reason!r}")
        exit_ticks = self.slippage_ticks_stop if exit_reason in MARKET_EXIT_REASONS else 0
        entry_ticks = self.slippage_ticks_entry if charge_entry else 0
        cost = self.commission_rt + (entry_ticks + exit_ticks) * self.tick_value
        return cost * self.multiplier

    def net_pnl(self, gross: float, exit_reason: str, charge_entry: bool = True) -> float:
        """Net P&L for ONE fill leg: `gross` minus this leg's cost (see
        `leg_cost`). Call once for a single-fill trade (charge_entry
        defaults True); call once PER LEG and sum for a multi-leg trade,
        passing `charge_entry=False` on every leg after the first."""
        return gross - self.leg_cost(exit_reason, charge_entry=charge_entry)
