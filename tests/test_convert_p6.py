import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data import convert_p6


class _Exploit:
    """Classic pickle RCE shape: __reduce__ returning (callable, args).
    Pickling this never executes os.system -- only *loading* it with a
    non-restricted unpickler would. We assert our RestrictedUnpickler
    refuses to even resolve the `os.system` global."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _dump(tmp_path, name, obj):
    p = tmp_path / name
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return str(p)


def test_converter_rejects_forbidden_global(tmp_path):
    src = _dump(tmp_path, "evil.pkl", [_Exploit()])
    with pytest.raises(pickle.UnpicklingError):
        convert_p6.load_candles_safely(src)


def test_converter_roundtrip_sorts_and_dedups(tmp_path, fake_candle_module):
    Candle = fake_candle_module
    base = datetime(2023, 1, 3, 9, 30)
    candles = [
        Candle(100.0, 101.0, 99.5, 100.5, base + timedelta(minutes=1)),  # out-of-order
        Candle(100.0, 100.5, 99.0, 100.0, base),
        Candle(100.0, 100.5, 99.0, 100.0, base),                         # exact dup (identical)
        Candle(101.0, 102.0, 100.5, 101.5, base + timedelta(minutes=2)),
        Candle(101.5, 102.5, 101.0, 102.0, base + timedelta(minutes=3)),
    ]
    src = _dump(tmp_path, "sample.pkl", candles)
    out = str(tmp_path / "sample.parquet")

    stats = convert_p6.convert(src, out)

    assert stats == {"n_in": 5, "n_out": 4, "dups_dropped": 1}

    df = pd.read_parquet(out)
    assert list(df.columns) == ["t", "o", "h", "l", "c"]
    ts = list(df["t"])
    assert ts == sorted(ts)
    assert df["t"].is_unique
    assert len(df) == 4

    dup_stats = convert_p6.read_dup_stats(out)
    assert dup_stats["n_dups_identical"] == 1
    assert dup_stats["n_dups_divergent"] == 0


def test_dedup_continuity_picks_price_consistent_row_not_keep_first(fake_candle_module):
    """A synthetic dup group where keep-first would pick a bad print, but the
    continuity rule (min |open - prev_kept_close|, tie -> keep-first) picks
    the row consistent with the preceding close."""
    Candle = fake_candle_module
    t0 = datetime(2023, 1, 3, 9, 30)
    dup_ts = t0 + timedelta(minutes=1)

    prior = Candle(99.5, 100.2, 99.0, 100.0, t0)                     # kept close = 100.0
    bad_first = Candle(150.0, 151.0, 149.0, 150.5, dup_ts)           # divergent; keep-first would pick this
    good_second = Candle(100.1, 100.6, 99.8, 100.4, dup_ts)          # near prev_close=100.0 -> continuity pick
    after = Candle(100.5, 101.0, 100.0, 100.8, t0 + timedelta(minutes=2))

    df = convert_p6.candles_to_frame([prior, bad_first, good_second, after])
    deduped, stats = convert_p6.dedup_continuity(df)

    assert stats["n_dups_divergent"] == 1
    assert stats["n_dups_identical"] == 0
    assert stats["dup_divergence_pctiles"]["p50"] == pytest.approx(150.5 - 100.4)

    kept_row = deduped.loc[deduped.index == pd.Timestamp(dup_ts)]
    assert len(kept_row) == 1
    assert kept_row["o"].iloc[0] == pytest.approx(100.1)   # continuity pick, NOT keep-first (150.0)


def test_dedup_continuity_tie_keeps_first():
    """Two divergent rows equidistant from prev_close -> tie -> keep-first."""
    t0 = pd.Timestamp("2023-01-03 09:30")
    dup_ts = pd.Timestamp("2023-01-03 09:31")
    df = pd.DataFrame(
        {
            "o": [100.0, 101.0, 99.0],   # dup group opens: 101.0 and 99.0, both |diff|=1.0 from prev_close=100.0
            "h": [100.5, 101.5, 99.5],
            "l": [99.5, 100.5, 98.5],
            "c": [100.0, 101.2, 99.2],
        },
        index=pd.DatetimeIndex([t0, dup_ts, dup_ts]),
    )
    deduped, stats = convert_p6.dedup_continuity(df)
    assert stats["n_dups_divergent"] == 1
    kept = deduped.loc[deduped.index == dup_ts]
    assert kept["o"].iloc[0] == pytest.approx(101.0)  # first candidate on a tie
