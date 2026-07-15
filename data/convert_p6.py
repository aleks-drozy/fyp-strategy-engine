"""Safe pickle -> parquet converter for the Phase-6 raw Candle data.

SECURITY NOTE: the raw ``data/raw_p6/{ES_2,NQ_2,YM_2}`` files are Python
pickles. ``pickle.load`` on an arbitrary/untrusted pickle can execute
arbitrary code (any GLOBAL/STACK_GLOBAL opcode can reference and then call
any importable callable, e.g. ``os.system``). We never call the stdlib
``pickle.load``/``pickle.loads`` directly on these files. Instead
``RestrictedUnpickler`` overrides ``find_class`` with an explicit allowlist:

  - ``("Candle", "Candle")``                      -> our local stub class
  - ``("numpy.core.multiarray", "scalar")``        -> real numpy (reconstructs a float64)
  - ``("numpy.core.multiarray", "_reconstruct")``  -> real numpy
  - ``("numpy", "dtype")``                         -> real numpy
  - ``("numpy", "ndarray")``                       -> real numpy
  - ("datetime", "datetime")``                    -> real stdlib datetime

Any other (module, name) pair -- e.g. ``os.system``, ``subprocess.Popen``,
``builtins.eval`` -- raises ``pickle.UnpicklingError`` *before* the referenced
object is ever resolved or called. This is a defense-in-depth measure: even
though this particular data is not attacker-supplied, treating an opaque
binary blob as hostile input is the only safe default for ``pickle``.

Parquet (written by ``convert()``) is the only artifact the rest of the
Phase-6 pipeline reads afterward -- the raw pickles and this restricted
unpickler are only ever touched once, here.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_DIR = "data/parquet_p6"
DUP_STATS_KEY = b"phase6_dup_stats"


class Candle:
    """Local stand-in for the source ``Candle`` class.

    The source pickles reference a class whose ``__module__`` and
    ``__qualname__`` are both literally ``"Candle"`` (i.e. a top-level class
    named ``Candle`` defined in a module named ``Candle``). We never import
    that module -- ``RestrictedUnpickler.find_class`` substitutes this local
    class whenever it sees that exact (module, name) pair.

    Instances are built via the default pickle protocol-2 object reduction
    (``Candle.__new__(Candle)`` then ``instance.__dict__.update(state)``),
    which never executes user code -- so a plain attribute bag is sufficient.
    """


_ALLOWED_REAL_GLOBALS = {
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("datetime", "datetime"),
}
_CANDLE_KEY = ("Candle", "Candle")


class RestrictedUnpickler(pickle.Unpickler):
    """``pickle.Unpickler`` whose ``find_class`` allows only the pairs above."""

    def find_class(self, module, name):
        key = (module, name)
        if key == _CANDLE_KEY:
            return Candle
        if key in _ALLOWED_REAL_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Forbidden global during restricted unpickle: {module}.{name}"
        )


def load_candles_safely(path: str) -> list:
    """Load the top-level list of ``Candle`` objects via the restricted unpickler."""
    with open(path, "rb") as fh:
        obj = RestrictedUnpickler(fh).load()
    if not isinstance(obj, list):
        raise pickle.UnpicklingError(
            f"expected a top-level list of Candle objects, got {type(obj)!r}"
        )
    return obj


def candles_to_frame(candles: list) -> pd.DataFrame:
    """Convert a list of Candle stubs into a naive-indexed OHLC frame.

    Sorted by timestamp with a *stable* sort, so rows that share a timestamp
    retain their original (file) order -- this is what makes "tie -> keep
    first" in ``dedup_continuity`` well-defined.
    """
    n = len(candles)
    o = np.empty(n, dtype="float64")
    h = np.empty(n, dtype="float64")
    l = np.empty(n, dtype="float64")
    c = np.empty(n, dtype="float64")
    t = [None] * n
    for i, cndl in enumerate(candles):
        o[i] = cndl.o
        h[i] = cndl.h
        l[i] = cndl.l
        c[i] = cndl.c
        t[i] = cndl.t
    idx = pd.DatetimeIndex(t, name="t")
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)
    return df.sort_index(kind="mergesort")  # mergesort == stable


def dedup_continuity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate-timestamp rows using the frozen price-continuity rule.

    ``df`` must already be sorted by index (stable sort) ascending; it may
    contain duplicate timestamps. Within each duplicate-timestamp group, the
    kept row is the one minimizing ``|open - previously_kept_close|``
    (continuity with whatever row precedes the group in the deduped output);
    ties keep the first row in (stable-sort / original file) order.

    Returns ``(deduped_df, stats)`` where ``stats`` carries
    ``n_in, n_out, dups_dropped, n_dups_identical, n_dups_divergent,
    dup_divergence_pctiles`` (percentiles of the close-price range within
    each *divergent* duplicate group).
    """
    n = len(df)
    if n == 0:
        return df.copy(), {
            "n_in": 0, "n_out": 0, "dups_dropped": 0,
            "n_dups_identical": 0, "n_dups_divergent": 0,
            "dup_divergence_pctiles": {},
        }

    t_vals = df.index.to_numpy().view("int64")
    opens = df["o"].to_numpy()
    highs = df["h"].to_numpy()
    lows = df["l"].to_numpy()
    closes = df["c"].to_numpy()

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = t_vals[1:] != t_vals[:-1]
    group_starts = np.flatnonzero(change)
    group_sizes = np.diff(np.append(group_starts, n))

    keep_mask = np.ones(n, dtype=bool)
    effective_close = closes.copy()  # close of the row that WOULD be kept at each position

    n_dups_identical = 0
    n_dups_divergent = 0
    divergences: list[float] = []

    multi = group_starts[group_sizes > 1]
    multi_sizes = group_sizes[group_sizes > 1]
    for start, size in zip(multi.tolist(), multi_sizes.tolist()):
        end = start + size
        go, gh, gl, gc = opens[start:end], highs[start:end], lows[start:end], closes[start:end]
        identical = (
            np.all(go == go[0]) and np.all(gh == gh[0])
            and np.all(gl == gl[0]) and np.all(gc == gc[0])
        )
        if identical:
            n_dups_identical += 1
            keep_local = 0
        else:
            n_dups_divergent += 1
            divergences.append(float(gc.max() - gc.min()))
            prev_close = effective_close[start - 1] if start > 0 else None
            if prev_close is None:
                keep_local = 0  # no continuity reference yet -> keep-first
            else:
                diffs = np.abs(go - prev_close)
                keep_local = int(np.flatnonzero(diffs == diffs.min())[0])  # tie -> keep-first
        keep_mask[start:end] = False
        keep_mask[start + keep_local] = True
        effective_close[start:end] = gc[keep_local]

    out = df[keep_mask].copy()
    pctiles: dict = {}
    if divergences:
        arr = np.asarray(divergences)
        pctiles = {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }
    stats = {
        "n_in": n,
        "n_out": int(keep_mask.sum()),
        "dups_dropped": int(n - keep_mask.sum()),
        "n_dups_identical": n_dups_identical,
        "n_dups_divergent": n_dups_divergent,
        "dup_divergence_pctiles": pctiles,
    }
    return out, stats


def parquet_path(sym: str, base_dir: str | None = None) -> str:
    return os.path.join(base_dir or PARQUET_DIR, f"{sym.upper()}.parquet")


def convert(src_path: str, out_parquet: str) -> dict:
    """Safely unpickle ``src_path``, dedup+sort, write parquet, return stats."""
    candles = load_candles_safely(src_path)
    raw = candles_to_frame(candles)
    deduped, stats = dedup_continuity(raw)

    out_dir = os.path.dirname(out_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frame = deduped.reset_index()  # index "t" -> column "t"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[DUP_STATS_KEY] = json.dumps(stats).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, out_parquet)

    return {"n_in": stats["n_in"], "n_out": stats["n_out"], "dups_dropped": stats["dups_dropped"]}


def read_dup_stats(parquet_path_: str) -> dict:
    """Read back the conversion-time dedup stats stashed in parquet schema metadata."""
    schema = pq.read_schema(parquet_path_)
    meta = schema.metadata or {}
    raw = meta.get(DUP_STATS_KEY)
    return json.loads(raw) if raw else {}


def main() -> None:
    results = {}
    for sym, fname in (("ES", "ES_2"), ("NQ", "NQ_2"), ("YM", "YM_2")):
        src = os.path.join("data", "raw_p6", fname)
        out = parquet_path(sym)
        print(f"converting {src} -> {out} ...")
        stats = convert(src, out)
        print(f"  {sym}: {stats}")
        results[sym] = stats
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
