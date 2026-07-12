"""Load + clean the raw NQ 1-min CSV into a canonical US/Eastern-indexed OHLCV DataFrame."""
from __future__ import annotations
import pandas as pd

DEFAULT_PATH = "data/raw/Dataset_NQ_1min_2022_2025.csv"
RENAME = {"Vwap_RTH": "vwap_rth", "Vwap_ETH": "vwap_eth"}
COLS = ["open", "high", "low", "close", "volume", "vwap_rth", "vwap_eth"]


def load_nq(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_PATH)
    ts = pd.to_datetime(df["timestamp ET"], errors="raise")
    idx = pd.DatetimeIndex(ts).tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    df = df.drop(columns=["timestamp ET"]).rename(columns=RENAME)
    df.index = idx
    df.index.name = "timestamp_et"
    df = df[df.index.notna()]                                  # drop DST-ambiguous NaT rows
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[COLS]


def session_slice(df: pd.DataFrame, start: str = "09:32", end: str = "10:00") -> pd.DataFrame:
    return df.between_time(start, end, inclusive="left")
