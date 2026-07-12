"""Validate the canonical NQ dataset; write a small committed report."""
from __future__ import annotations
import json
import pandas as pd
from nqdata.load import load_nq, session_slice


def validate_nq(df: pd.DataFrame) -> dict:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ohlc_viol = int(((h < o) | (h < c) | (l > o) | (l > c) | (h < l)).sum())
    nan_ohlc = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    spacing = df.index.to_series().diff().dropna()
    pct_1min = float((spacing == pd.Timedelta(minutes=1)).mean()) if len(spacing) else 0.0
    sess = session_slice(df)
    return {
        "n_rows": int(len(df)),
        "date_min": str(df.index.min()),
        "date_max": str(df.index.max()),
        "timezone": str(df.index.tz),
        "pct_1min_spacing": pct_1min,
        "n_ohlc_violations": ohlc_viol,
        "n_nan_ohlc": nan_ohlc,
        "n_dup_timestamps": int(df.index.duplicated().sum()),
        "session_bar_count": int(len(sess)),
        "session_days": int(sess.index.normalize().nunique()),
    }


def main() -> None:
    report = validate_nq(load_nq())
    with open("data/validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
