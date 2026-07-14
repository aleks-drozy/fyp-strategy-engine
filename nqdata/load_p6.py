"""Load a Phase-6 per-instrument parquet into the canonical OHLCV contract.

Same contract as ``nqdata.load.load_nq``: lowercase o/h/l/c columns (+
synthesized ``volume=0`` -- the source Candle objects carry no volume),
tz-aware US/Eastern DatetimeIndex, sorted, deduped.

Two frozen, disclosed data-cleaning rules are applied here (both decided
before any backtest touches this data -- see ``validation_report_p6.json``
for the evidence that justifies each):

1. **Bar-labeling normalization** (``BAR_LABEL_SHIFT_MINUTES``): the Kaggle
   vendor stamps 1-min bars on a different convention than Phase-1's
   ``load_nq()`` (open-time vs close-time). The correct shift was determined
   EMPIRICALLY (see ``validate_p6.compute_bar_label_shift_evidence`` -- the
   per-day cross-vendor log-return correlation at candidate shifts
   {-1, 0, +1} min over the 2023 NQ overlap is decisive for +1 min: median
   daily corr ~0.90 vs ~0.01 and ~0.00 for the other two). Applied uniformly
   to all three instruments (same vendor, same convention) to the NAIVE
   timestamp, before tz localization.
2. **Maintenance-hour drop** (``MAINTENANCE_HOUR_START/END``): CME's daily
   17:00-18:00 ET maintenance break is when the market is closed -- no real
   bars should exist there. In 2020 through mid-2021 the vendor spliced an
   unrelated series into this dead hour for ES/NQ (a real data defect, not a
   trading signal); YM is clean. Rather than special-case the contaminated
   years, every bar with ET local time in [17:00, 18:00) is dropped for
   every instrument, uniformly -- market-structure-grounded, not tuned to
   the defect. Applied AFTER the shift normalization (so the drop window is
   defined in TRUE wall-clock ET, not the vendor's mislabeled time).

``load_instrument()`` (shift + drop applied) is the contract every
downstream consumer (engine, backtests) must use. ``load_instrument_raw()``
(shift applied, drop NOT applied) and ``load_instrument_unshifted()`` (no
shift, no drop -- the literal vendor timestamps) exist only for
``validate_p6.py`` to compute evidence about the two cleaning rules
themselves; nothing downstream of validation should ever call them.
"""
from __future__ import annotations

import pandas as pd

from data.convert_p6 import parquet_path

COLS = ["open", "high", "low", "close", "volume"]
_VALID_SYMS = ("ES", "NQ", "YM")

# --- Defect 2: bar-labeling normalization (frozen; see module docstring) ---
BAR_LABEL_SHIFT_MINUTES = 1

# --- Defect 1: maintenance-hour drop (frozen; see module docstring) --------
MAINTENANCE_HOUR_START = 17
MAINTENANCE_HOUR_END = 18  # half-open: drop ET local time in [17:00, 18:00)


def _read_localized(sym: str, path: str | None, shift_minutes: int) -> pd.DataFrame:
    sym_u = sym.upper()
    if sym_u not in _VALID_SYMS:
        raise ValueError(f"unknown instrument {sym!r}; expected one of {_VALID_SYMS}")

    df = pd.read_parquet(path or parquet_path(sym_u), columns=["t", "o", "h", "l", "c"])
    naive = pd.DatetimeIndex(df["t"])
    if shift_minutes:
        naive = naive + pd.Timedelta(minutes=shift_minutes)
    idx = naive.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
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


def drop_maintenance_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars with ET local time in [17:00, 18:00) -- the CME daily
    maintenance break, when the market is closed. Frozen data-cleaning rule
    (Defect 1); see module docstring."""
    hour = df.index.hour
    return df[~((hour >= MAINTENANCE_HOUR_START) & (hour < MAINTENANCE_HOUR_END))]


def load_instrument_unshifted(sym: str, path: str | None = None) -> pd.DataFrame:
    """The literal vendor timestamps: no bar-label shift, no maintenance-hour
    drop. Used ONLY by ``validate_p6.compute_bar_label_shift_evidence`` to
    empirically re-derive/document the shift choice -- never for downstream
    consumption."""
    return _read_localized(sym, path, shift_minutes=0)


def load_instrument_raw(sym: str, path: str | None = None) -> pd.DataFrame:
    """Shift-normalized but WITHOUT the maintenance-hour drop. Used ONLY by
    ``validate_p6`` to compute tz evidence and contamination evidence on the
    pre-cleaning bars -- never for downstream consumption."""
    return _read_localized(sym, path, shift_minutes=BAR_LABEL_SHIFT_MINUTES)


def load_instrument(sym: str, path: str | None = None) -> pd.DataFrame:
    """Canonical Phase-6 contract: bar-label shift + maintenance-hour drop
    applied. This is what every downstream consumer must use."""
    return drop_maintenance_hour(load_instrument_raw(sym, path))
