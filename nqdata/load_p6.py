"""Load a Phase-6 per-instrument parquet into the canonical OHLCV contract.

Same contract as ``nqdata.load.load_nq``: lowercase o/h/l/c columns (+
synthesized ``volume=0`` -- the source Candle objects carry no volume),
tz-aware US/Eastern DatetimeIndex, sorted, deduped.
"""
from __future__ import annotations

import pandas as pd

from data.convert_p6 import parquet_path

COLS = ["open", "high", "low", "close", "volume"]
_VALID_SYMS = ("ES", "NQ", "YM")


def load_instrument(sym: str, path: str | None = None) -> pd.DataFrame:
    sym_u = sym.upper()
    if sym_u not in _VALID_SYMS:
        raise ValueError(f"unknown instrument {sym!r}; expected one of {_VALID_SYMS}")

    df = pd.read_parquet(path or parquet_path(sym_u), columns=["t", "o", "h", "l", "c"])
    idx = pd.DatetimeIndex(df["t"]).tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
    )
    out = pd.DataFrame(
        {
            "open": df["o"].to_numpy(),
            "high": df["h"].to_numpy(),
            "low": df["l"].to_numpy(),
            "close": df["c"].to_numpy(),
            "volume": 0,
        },
        index=idx,
    )
    out.index.name = "timestamp_et"
    out = out[out.index.notna()]                              # drop DST-ambiguous NaT rows
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out[COLS]
