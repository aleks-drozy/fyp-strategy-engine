"""Per-instrument contract specifications (Phase 6).

Frozen constants from the Phase-6 plan's Global Constraints — CME e-mini
contract specs, used to thread $/point and $/tick through the engine and
cost model so ES/NQ/YM are each costed and P&L'd correctly:

  ES: tick 0.25 = $12.50  ->  $50/pt
  NQ: tick 0.25 = $5.00   ->  $20/pt   (the Phase 1-5 default, regression-locked)
  YM: tick 1.00 = $5.00   ->  $5/pt

`CostModel` already carries `tick_value` as a field; `for_instrument`
builds the per-instrument cost model with the Phase-5 pre-registered cost
structure unchanged ($5 RT commission, 1 tick entry + 1 tick per market
exit — only the tick's dollar value varies by instrument).
"""

from dataclasses import dataclass

from backtest.costs import CostModel


@dataclass(frozen=True)
class InstrumentSpec:
    sym: str
    tick_size: float
    tick_value: float
    pt_value: float

    def cost_model(self, multiplier: float = 1.0) -> CostModel:
        """The Phase-5 cost structure valued at this instrument's tick."""
        return CostModel(tick_value=self.tick_value, multiplier=multiplier)


SPECS = {
    "ES": InstrumentSpec(sym="ES", tick_size=0.25, tick_value=12.50, pt_value=50.0),
    "NQ": InstrumentSpec(sym="NQ", tick_size=0.25, tick_value=5.00, pt_value=20.0),
    "YM": InstrumentSpec(sym="YM", tick_size=1.00, tick_value=5.00, pt_value=5.0),
}
