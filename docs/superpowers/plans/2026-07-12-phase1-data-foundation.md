# Phase 1 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible pipeline that downloads the Kaggle NQ 1-minute CSV and turns it into a canonical, validated, ET-indexed OHLCV dataset the Phase 2 backtest can trust.

**Architecture:** `data/download.py` fetches the Kaggle dataset (key from `~/.kaggle/`, never in repo; raw gitignored). `nqdata/load.py` parses it into a clean US/Eastern-indexed OHLCV DataFrame + a session helper. `validate.py` checks OHLC/timezone/spacing/session invariants and writes a committed `data/validation_report.json`.

**Tech Stack:** Python 3.11+, pandas, kaggle, numpy, pytest.

## Global Constraints

- Python **3.11+**; deps pinned in `requirements.txt` (`pandas`, `kaggle`, `numpy`, `pytest`).
- **Kaggle key** read from `~/.kaggle/access_token` — **never** committed. `data/raw/` is **git-ignored** (72 MB, Kaggle's data). Only the small `data/validation_report.json` + code are committed.
- **Timezone:** timestamps parsed as **US/Eastern**; canonical columns `open, high, low, close, volume` (+ `vwap_rth, vwap_eth`), index name `timestamp_et`.
- **Cleaning:** sort ascending, drop exact-duplicate timestamps (keep first), drop DST-ambiguous NaT rows. Keep ALL bars (no session filtering in the loader).
- **Session window:** `[09:32, 10:00)` ET via `session_slice`.
- **Windows:** venv Python `.venv/Scripts/python`; commit on `master` (solo new repo); never `git add` `.venv/` or `.superpowers/` or `data/raw/`.

---

### Task 1: Scaffold + toolchain + download script

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `README.md`
- Create: `data/__init__.py`, `nqdata/__init__.py`, `tests/__init__.py` (empty)
- Create: `data/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces: `download_nq(dest="data/raw") -> str` (returns the CSV path; idempotent — returns the cached path without hitting Kaggle if the file already exists). `DATASET`, `CSV_NAME` constants.

- [ ] **Step 1: Write `requirements.txt`**

```
pandas==2.2.3
numpy==2.1.3
kaggle==2.2.3
pytest==8.3.3
```
(kaggle **2.2.3** — the version verified in the spike to read `~/.kaggle/access_token`; the older 1.6.x only supports `kaggle.json` and would fail auth here.)

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
data/raw/
*.zip
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
pythonpath = .
```

- [ ] **Step 4: Create packages + README** — empty `data/__init__.py`, `nqdata/__init__.py`, `tests/__init__.py`; `README.md` with `# fyp-strategy-engine` and a one-line note ("Rebuilding + extending the FYP IFVG+CISD NQ strategy in Python. Phase 1: data foundation.").

- [ ] **Step 5: Write the failing test** `tests/test_download.py` (idempotent-skip path — no network)

```python
from data.download import download_nq, CSV_NAME

def test_download_is_idempotent_when_cached(tmp_path):
    # pre-create the expected CSV so download_nq returns it WITHOUT calling kaggle
    (tmp_path / CSV_NAME).write_text("timestamp ET,open,high,low,close,volume,Vwap_RTH,Vwap_ETH\n")
    p = download_nq(dest=str(tmp_path))
    assert p.endswith(CSV_NAME)
    import os
    assert os.path.exists(p)
```

- [ ] **Step 6: Run to verify failure** — `.venv/Scripts/python -m pytest tests/test_download.py -q` → FAIL (`ModuleNotFoundError: No module named 'data.download'`).

- [ ] **Step 7: Implement `data/download.py`**

```python
"""Download the Kaggle NQ 1-min dataset. Key read from ~/.kaggle/access_token; never stored here."""
from __future__ import annotations
import os

DATASET = "tgtanalytics/nq-futures-1min-bar-2022-2025"
CSV_NAME = "Dataset_NQ_1min_2022_2025.csv"


def download_nq(dest: str = "data/raw") -> str:
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, CSV_NAME)
    if os.path.exists(path):
        return path                      # idempotent: skip the fetch if already cached
    import kaggle                        # authenticates on import via ~/.kaggle/access_token
    kaggle.api.dataset_download_files(DATASET, path=dest, unzip=True)
    if not os.path.exists(path):
        raise RuntimeError(f"download did not produce {path}")
    return path


def main() -> None:
    p = download_nq()
    print(f"NQ data at {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Create venv, install, run test** — `python -m venv .venv` → `.venv/Scripts/python -m pip install -r requirements.txt` → `.venv/Scripts/python -m pytest -q` → `1 passed`.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .gitignore pytest.ini README.md data/__init__.py nqdata/__init__.py tests/__init__.py data/download.py tests/test_download.py
git commit -m "chore: scaffold fyp-strategy-engine + idempotent Kaggle NQ downloader"
```

---

### Task 2: Loader + session helper

**Files:**
- Create: `nqdata/load.py`
- Test: `tests/test_load.py`

**Interfaces:**
- Produces: `load_nq(path: str | None = None) -> pandas.DataFrame` (US/Eastern DatetimeIndex named `timestamp_et`; columns `open, high, low, close, volume, vwap_rth, vwap_eth`; sorted; deduped). `session_slice(df, start="09:32", end="10:00") -> pandas.DataFrame`.

- [ ] **Step 1: Write the failing tests** `tests/test_load.py`

```python
from nqdata.load import load_nq, session_slice

def _fixture(tmp_path):
    csv = tmp_path / "nq.csv"
    csv.write_text(
        "timestamp ET,open,high,low,close,volume,Vwap_RTH,Vwap_ETH\n"
        "01/03/2023 09:33,100,101,99.5,100.5,10,0,100.2\n"
        "01/03/2023 09:32,100,100.5,99,100,12,0,99.8\n"        # out of order
        "01/03/2023 09:32,100,100.5,99,100,12,0,99.8\n"        # duplicate timestamp
        "01/03/2023 10:05,101,102,100.5,101.5,8,101,101.1\n"
    )
    return str(csv)

def test_load_parses_tz_renames_sorts_dedups(tmp_path):
    df = load_nq(_fixture(tmp_path))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap_rth", "vwap_eth"]
    assert str(df.index.tz) == "US/Eastern"
    assert df.index.name == "timestamp_et"
    assert df.index.is_monotonic_increasing
    assert len(df) == 3                                        # one duplicate 09:32 dropped
    assert df.index[0].strftime("%H:%M") == "09:32"

def test_session_slice_window(tmp_path):
    df = load_nq(_fixture(tmp_path))
    s = session_slice(df, "09:32", "10:00")
    assert len(s) == 2                                         # 09:32 + 09:33; 10:05 excluded
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python -m pytest tests/test_load.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `nqdata/load.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass** — `.venv/Scripts/python -m pytest tests/test_load.py -q` → `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add nqdata/load.py tests/test_load.py
git commit -m "feat: canonical ET-indexed NQ loader + session slice"
```

---

### Task 3: Validator + real-data validation report

**Files:**
- Create: `validate.py`
- Create: `data/validation_report.json` (generated from the REAL data, committed)
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `nqdata.load` (load_nq, session_slice).
- Produces: `validate_nq(df) -> dict` (keys: `n_rows, date_min, date_max, timezone, pct_1min_spacing, n_ohlc_violations, n_nan_ohlc, n_dup_timestamps, session_bar_count, session_days`); `main()` writes `data/validation_report.json`.

- [ ] **Step 1: Write the failing tests** `tests/test_validate.py`

```python
from nqdata.load import load_nq
from validate import validate_nq

def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("timestamp ET,open,high,low,close,volume,Vwap_RTH,Vwap_ETH\n" + body)
    return str(p)

def test_validate_flags_ohlc_violation(tmp_path):
    # row 2: high(99) < open(100) and < close(101) -> OHLC violation
    path = _write(tmp_path, "bad.csv",
                  "01/03/2023 09:32,100,100.5,99,100,10,0,100\n"
                  "01/03/2023 09:33,100,99,99,101,10,0,100\n")
    r = validate_nq(load_nq(path))
    assert r["n_ohlc_violations"] >= 1
    assert set(r) >= {"n_rows", "date_min", "date_max", "timezone", "pct_1min_spacing",
                      "n_ohlc_violations", "n_nan_ohlc", "n_dup_timestamps",
                      "session_bar_count", "session_days"}

def test_validate_clean_fixture(tmp_path):
    path = _write(tmp_path, "clean.csv",
                  "01/03/2023 09:32,100,101,99,100.5,10,0,100\n"
                  "01/03/2023 09:33,100.5,101.5,100,101,12,0,100.5\n")
    r = validate_nq(load_nq(path))
    assert r["n_ohlc_violations"] == 0 and r["n_nan_ohlc"] == 0
    assert r["timezone"] == "US/Eastern"
    assert r["session_bar_count"] == 2
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python -m pytest tests/test_validate.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `validate.py`**

```python
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
```

- [ ] **Step 4: Run the tests, then the REAL pipeline**

Run: `.venv/Scripts/python -m pytest tests/test_validate.py -q` → `2 passed`; then `.venv/Scripts/python -m pytest -q` → all pass.
Then generate the real report (needs the Kaggle key at `~/.kaggle/access_token`, already set up):
`.venv/Scripts/python data/download.py` (downloads the 72 MB CSV to `data/raw/` if not cached), then `.venv/Scripts/python validate.py`.
Expected: `data/validation_report.json` written. **Read it and confirm the honest checks:** `n_ohlc_violations` ≈ 0, `n_nan_ohlc` == 0, `pct_1min_spacing` > 0.99, `timezone` == "US/Eastern", `date_min` ~2022-12, `date_max` ~2025-12, `session_days` in the hundreds. Record these in the report file below.

- [ ] **Step 5: Commit** (the report is committed; the raw 72 MB CSV stays git-ignored)

```bash
git add validate.py tests/test_validate.py data/validation_report.json
git commit -m "feat: NQ dataset validator + committed real-data validation report"
```

---

## Notes for the implementer

- **Windows shell:** `.venv/Scripts/python`; shell snippets in Git Bash. Repo `C:\Users\Alex\Projects\fyp-strategy-engine`; commit on `master`; never `git add` `.venv/`, `.superpowers/`, or `data/raw/`.
- **The Kaggle key** lives at `~/.kaggle/access_token` (already set up + tested). It must never enter the repo or a commit; only `data/download.py`'s runtime references it.
- **Honesty:** record the real numbers in `data/validation_report.json` as they come out — including the Dec-2025 truncation (date_max). Do not fabricate coverage the data doesn't have.
- **Next (out of scope):** Phase 2 reimplements IFVG+CISD in Python over this dataset and validates the generated trades against the real 167 FYP trades.
