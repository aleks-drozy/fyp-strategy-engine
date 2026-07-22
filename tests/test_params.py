"""Tests for strategy.params.StrategyParams.

From the Phase-4 parameter-tuning spec
(docs/specs/2026-07-13-phase4-parameter-tuning-design.md). Asserts the
documented defaults (which MUST reproduce Phase 2 exactly) and that the
dataclass is frozen (hashable, assignment raises).

The Phase-5 spec
(docs/specs/2026-07-13-phase5-exits-costs-volfilter-design.md) adds
`exit_mode`/`vol_filter`, both defaulting to the Phase-2/4 base behavior.
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


def test_phase5_defaults_are_behavior_preserving():
    p = StrategyParams()
    assert p.exit_mode == "fixed_1_5R"
    assert p.vol_filter == "off"


def test_phase5_fields_are_overridable():
    p = StrategyParams(exit_mode="trail_swing", vol_filter="p50")
    assert p.exit_mode == "trail_swing"
    assert p.vol_filter == "p50"


def test_is_frozen_and_hashable():
    p = StrategyParams()
    # frozen dataclasses are hashable by default; this raises TypeError if not
    hash(p)

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.rr = 2.0
