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
