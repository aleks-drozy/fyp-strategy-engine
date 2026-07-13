import pandas as pd
from strategy.signals import double_confirmation


def _s(vals):
    return pd.Series(vals)


def test_long_fires_only_on_cisd_flip_bar_when_ifvg_bullish():
    # bar2: cisd flips Bearish->Bullish while ifvg already Bullish => Long on bar2 only
    ifvg = _s(["Bullish", "Bullish", "Bullish", "Bullish"])
    cisd = _s(["Bearish", "Bearish", "Bullish", "Bullish"])
    out = double_confirmation(ifvg, cisd).tolist()
    assert out == ["", "", "Long", ""]


def test_no_fire_on_static_alignment():
    ifvg = _s(["Bullish", "Bullish", "Bullish"])
    cisd = _s(["Bullish", "Bullish", "Bullish"])  # already aligned, no flip
    assert double_confirmation(ifvg, cisd).tolist() == ["", "", ""]


def test_short_mirror():
    ifvg = _s(["Bearish", "Bearish", "Bearish"])
    cisd = _s(["Bullish", "Bullish", "Bearish"])  # flip to Bearish on bar2
    assert double_confirmation(ifvg, cisd).tolist() == ["", "", "Short"]


def test_ifvg_turns_and_cisd_flips_same_bar():
    ifvg = _s(["None", "None", "Bullish"])
    cisd = _s(["Bearish", "Bearish", "Bullish"])  # both turn bullish on bar2
    assert double_confirmation(ifvg, cisd).tolist() == ["", "", "Long"]


def test_short_ifvg_turns_and_cisd_flips_same_bar():
    # bear-side OR-branch 2: ifvg and cisd both turn bearish on bar2
    ifvg = _s(["None", "None", "Bearish"])
    cisd = _s(["Bullish", "Bullish", "Bearish"])
    assert double_confirmation(ifvg, cisd).tolist() == ["", "", "Short"]
