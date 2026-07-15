import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data import convert_p6
from nqdata.load_p6 import (
    BAR_LABEL_SHIFT_MINUTES,
    MAINTENANCE_HOUR_END,
    MAINTENANCE_HOUR_START,
    drop_maintenance_hour,
    load_instrument,
    load_instrument_raw,
    load_instrument_unshifted,
)


def _make_parquet(tmp_path, Candle, candles, name="nq"):
    src = tmp_path / f"{name}.pkl"
    with open(src, "wb") as fh:
        pickle.dump(candles, fh)
    out = tmp_path / f"{name}.parquet"
    convert_p6.convert(str(src), str(out))
    return str(out)


def _basic_candles(Candle):
    base = datetime(2023, 1, 3, 9, 30)
    return [
        Candle(100.5, 101.0, 100.0, 100.8, base + timedelta(minutes=1)),  # out-of-order on purpose
        Candle(100.0, 100.5, 99.0, 100.0, base),
        Candle(101.0, 101.5, 100.5, 101.2, base + timedelta(minutes=2)),
    ]


def test_load_instrument_contract(tmp_path, fake_candle_module):
    path = _make_parquet(tmp_path, fake_candle_module, _basic_candles(fake_candle_module))
    df = load_instrument("NQ", path=path)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "US/Eastern"
    assert df.index.name == "timestamp_et"
    assert df.index.is_monotonic_increasing
    assert (df["volume"] == 0).all()
    assert len(df) == 3
    # out-of-order row correctly sorted first; bar-label shift (+1 min,
    # Defect 2) applied: raw 09:30 -> loaded 09:31.
    assert BAR_LABEL_SHIFT_MINUTES == 1
    assert df.index[0].strftime("%H:%M") == "09:31"


def test_load_instrument_unshifted_has_no_bar_label_shift(tmp_path, fake_candle_module):
    path = _make_parquet(tmp_path, fake_candle_module, _basic_candles(fake_candle_module))
    df = load_instrument_unshifted("NQ", path=path)
    assert df.index[0].strftime("%H:%M") == "09:30"  # literal vendor timestamp, no shift


def test_load_instrument_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        load_instrument("CL")


@pytest.mark.parametrize("sym", ["es", "Nq", "YM"])
def test_load_instrument_case_insensitive_symbol(tmp_path, fake_candle_module, sym):
    path = _make_parquet(tmp_path, fake_candle_module, _basic_candles(fake_candle_module))
    df = load_instrument(sym, path=path)
    assert len(df) == 3


# --- Defect 1: maintenance-hour drop ----------------------------------------

def _maintenance_window_candles(Candle):
    """Bars every minute from 16:55 to 18:05 ET (post-shift wall clock), i.e.
    raw vendor timestamps from 16:54 to 18:04 (shift +1 -> +1 min later)."""
    base = datetime(2023, 1, 3, 16, 54)
    return [Candle(100.0, 100.1, 99.9, 100.0, base + timedelta(minutes=i)) for i in range(71)]


def test_load_instrument_drops_maintenance_hour(tmp_path, fake_candle_module):
    path = _make_parquet(tmp_path, fake_candle_module, _maintenance_window_candles(fake_candle_module), name="maint")
    raw = load_instrument_raw("NQ", path=path)
    clean = load_instrument("NQ", path=path)

    # raw (shift applied, drop not applied) still has bars inside [17:00, 18:00)
    assert ((raw.index.hour >= MAINTENANCE_HOUR_START) & (raw.index.hour < MAINTENANCE_HOUR_END)).any()
    # cleaned has none
    assert not ((clean.index.hour >= MAINTENANCE_HOUR_START) & (clean.index.hour < MAINTENANCE_HOUR_END)).any()
    # bars outside the window are untouched
    assert (clean.index.hour == 16).any()
    assert (clean.index.hour == 18).any()
    assert len(clean) < len(raw)


def test_drop_maintenance_hour_is_half_open():
    idx = pd.date_range("2023-01-03 16:58", "2023-01-03 18:02", freq="1min", tz="US/Eastern")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}, index=idx)
    out = drop_maintenance_hour(df)
    # 16:58, 16:59 kept; 17:00..17:59 dropped; 18:00, 18:01, 18:02 kept (18:00 boundary INCLUDED, not dropped)
    kept_times = set(out.index.strftime("%H:%M"))
    assert "16:59" in kept_times
    assert "17:00" not in kept_times
    assert "17:59" not in kept_times
    assert "18:00" in kept_times


def test_load_instrument_unshifted_vs_raw_vs_clean_ordering(tmp_path, fake_candle_module):
    """Sanity check the full pipeline ordering: unshifted has the literal
    vendor time; raw has the shift but not the drop; clean has both."""
    path = _make_parquet(tmp_path, fake_candle_module, _maintenance_window_candles(fake_candle_module), name="order")
    unshifted = load_instrument_unshifted("NQ", path=path)
    raw = load_instrument_raw("NQ", path=path)
    clean = load_instrument("NQ", path=path)

    assert len(unshifted) == len(raw) == 71
    assert len(clean) < len(raw)
    # raw is exactly unshifted's index + BAR_LABEL_SHIFT_MINUTES
    assert (raw.index == unshifted.index + pd.Timedelta(minutes=BAR_LABEL_SHIFT_MINUTES)).all()
