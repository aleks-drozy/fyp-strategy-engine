"""Tests for strategy.params.StrategyParams.

Phase 4, Task 1 Step 6 (docs/superpowers/plans/2026-07-13-phase4-parameter-tuning.md).
Asserts the documented defaults (Global Constraints -- MUST reproduce Phase 2
exactly) and that the dataclass is frozen (hashable, assignment raises).
"""

import dataclasses

import pytest

from strategy.params import StrategyParams


def test_defaults_match_phase2():
    p = StrategyParams()
    assert p.fvg_threshold == 0.0
    assert p.rr == 1.5
    assert p.ema_length == 20
    assert p.swing_lookback == 8
    assert p.session_start == "09:30"
    assert p.session_end == "10:30"


def test_is_frozen_and_hashable():
    p = StrategyParams()
    # frozen dataclasses are hashable by default; this raises TypeError if not
    hash(p)

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.rr = 2.0
