# Phase 1 — Data Foundation — Design Spec

**Date:** 2026-07-12
**Owner:** Aleksandrs Drozdovs
**Status:** approved (brainstorming), pending spec review

## Purpose

A reproducible pipeline that turns the Kaggle NQ 1-minute CSV into a **canonical, validated,
analysis-ready 1-minute NQ dataset (Eastern Time)** that the Phase 2 backtest can trust. This is the
first sub-project of the `fyp-strategy-engine` program — rebuilding and extending the FYP IFVG+CISD
strategy in Python because no TradingView premium is available to export more trades.

## Context

The FYP strategy is **intraday**: 1-minute bars, the 09:32–10:00 **New York** session. Without TV premium
the only way to get more results is to rebuild the strategy in Python over free data. A spike confirmed the
Kaggle dataset **`tgtanalytics/nq-futures-1min-bar-2022-2025`** is usable: real NQ 1-min bars, timestamps
already in **ET**, 1,048,575 rows, **Dec 2022 → Dec 2025** — which overlaps both real FYP trade periods, so
Phase 2 can validate the reimplementation against the real 167 trades. The file is truncated at Excel's row
limit (ends Dec 2025), so the last ~2.5 months of the winning period aren't covered — recorded honestly, not
hidden.

## Goals

1. **Reproducible download** via the Kaggle API — key read from `~/.kaggle/access_token`, **never** in the repo.
2. A **clean, validated canonical dataset** (ET tz, OHLC invariants enforced, 1-min, de-duplicated).
3. A **committed validation report** that proves the data is clean *without* committing the 72 MB raw file.
4. A clean loader + session helper that Phase 2 imports.

## Non-Goals

- No strategy logic (Phase 2). No look-ahead handling (the backtest's concern).
- No supplementing the Dec-2025→Feb-2026 tail here (noted; Dukascopy top-up is a later option).

## Data Source

Kaggle `tgtanalytics/nq-futures-1min-bar-2022-2025`, single file `Dataset_NQ_1min_2022_2025.csv`
(72 MB, 1,048,575 rows). Columns: `timestamp ET, open, high, low, close, volume, Vwap_RTH, Vwap_ETH`.
Timestamps are `MM/DD/YYYY HH:MM` in US/Eastern; bar spacing is 1 minute (with the normal daily
maintenance-hour break and weekend gaps).

## Components

1. **`data/download.py`** — `download_nq(dest="data/raw") -> str` (returns the CSV path). Uses the Kaggle
   API (reads `~/.kaggle/access_token`). **Idempotent:** skip the download if the CSV already exists.
   Raw CSV → `data/raw/` (git-ignored). `main()` prints the path + size.
2. **`nqdata/__init__.py`**, **`nqdata/load.py`**:
   - `load_nq(path: str | None = None) -> pandas.DataFrame` — read the raw CSV (default `data/raw/Dataset_NQ_1min_2022_2025.csv`),
     parse `timestamp ET` as a **US/Eastern-localized** `DatetimeIndex`, rename to `open, high, low, close, volume`
     (keep `vwap_rth, vwap_eth`), sort ascending, drop exact-duplicate timestamps (keep first). Returns a clean
     OHLCV DataFrame indexed by ET timestamp.
   - `session_slice(df, start="09:32", end="10:00") -> pandas.DataFrame` — rows whose ET time-of-day is within
     `[start, end)` (the strategy's session). Cleaning keeps **all** bars — the strategy needs pre-session
     history for IFVG/CISD/swing detection.
3. **`validate.py`**:
   - `validate_nq(df) -> dict` — returns a report: `n_rows, date_min, date_max, timezone, pct_1min_spacing,
     n_ohlc_violations, n_nan_ohlc, n_dup_timestamps, session_days, session_bar_count`.
   - `main()` — `load_nq()` → `validate_nq()` → write `data/validation_report.json` (committed).

## Validation checks (what the report asserts)

- **OHLC invariants:** `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` → violation count (≈ 0).
- **No NaN** in open/high/low/close/volume.
- **1-minute spacing dominates** (`pct_1min_spacing > 0.99`).
- **Timezone is US/Eastern.**
- **Date range** covers ~2023-01 → 2025-12.
- **Session populated:** the 09:32–10:00 ET window has bars on the large majority of trading days.

## Repo Layout

```
fyp-strategy-engine/
  data/{download.py, raw/ (gitignored), validation_report.json}
  nqdata/{__init__.py, load.py}
  validate.py
  tests/{test_load.py, test_validate.py}
  requirements.txt  README.md  .gitignore  pytest.ini
```

## Testing (TDD)

- **`test_load.py`** — on a tiny synthetic CSV fixture (a handful of ET rows, one duplicate, one out-of-order):
  `load_nq` parses the ET tz, renames columns, sorts, and drops the duplicate; `session_slice` returns only
  the 09:32–10:00 rows.
- **`test_validate.py`** — `validate_nq` flags a deliberately-broken OHLC row (high < close) and a NaN;
  passes on a clean fixture; the report dict has all expected keys.
- The real 72 MB dataset is **not** used in unit tests (tiny fixtures only); a separate smoke step runs the
  real `download` + `validate` once and commits `data/validation_report.json`.

## Reproducibility / dependencies

`requirements.txt`: `pandas`, `kaggle`, `pytest` (+ `numpy`). Nothing stochastic. Anyone with a Kaggle key runs
`python data/download.py && python validate.py` to regenerate the raw data + report. The key stays at
`~/.kaggle/` and is never committed; `data/raw/` is git-ignored.

## Risks

- **Kaggle key required** for the download (user supplies it; not in repo). The loader/validator work on the
  cached raw file, so only the initial fetch needs the key.
- **Excel truncation** (ends Dec 2025) → recorded in the report; the validation window still covers 2023–2025.
- **Dataset could change/vanish on Kaggle** — the committed `validation_report.json` captures the snapshot we
  validated (row count, date range, checksums-of-record), so drift is detectable.
