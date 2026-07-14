import pickle
from datetime import datetime, timedelta

import pytest

from data import convert_p6
from nqdata.load_p6 import load_instrument


def _make_parquet(tmp_path, Candle):
    base = datetime(2023, 1, 3, 9, 30)
    candles = [
        Candle(100.5, 101.0, 100.0, 100.8, base + timedelta(minutes=1)),  # out-of-order on purpose
        Candle(100.0, 100.5, 99.0, 100.0, base),
        Candle(101.0, 101.5, 100.5, 101.2, base + timedelta(minutes=2)),
    ]
    src = tmp_path / "nq.pkl"
    with open(src, "wb") as fh:
        pickle.dump(candles, fh)
    out = tmp_path / "nq.parquet"
    convert_p6.convert(str(src), str(out))
    return str(out)


def test_load_instrument_contract(tmp_path, fake_candle_module):
    path = _make_parquet(tmp_path, fake_candle_module)
    df = load_instrument("NQ", path=path)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "US/Eastern"
    assert df.index.name == "timestamp_et"
    assert df.index.is_monotonic_increasing
    assert (df["volume"] == 0).all()
    assert len(df) == 3
    assert df.index[0].strftime("%H:%M") == "09:30"  # out-of-order row correctly sorted first


def test_load_instrument_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        load_instrument("CL")


@pytest.mark.parametrize("sym", ["es", "Nq", "YM"])
def test_load_instrument_case_insensitive_symbol(tmp_path, fake_candle_module, sym):
    path = _make_parquet(tmp_path, fake_candle_module)
    df = load_instrument(sym, path=path)
    assert len(df) == 3
