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
