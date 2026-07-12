from data.download import download_nq, CSV_NAME

def test_download_is_idempotent_when_cached(tmp_path):
    # pre-create the expected CSV so download_nq returns it WITHOUT calling kaggle
    (tmp_path / CSV_NAME).write_text("timestamp ET,open,high,low,close,volume,Vwap_RTH,Vwap_ETH\n")
    p = download_nq(dest=str(tmp_path))
    assert p.endswith(CSV_NAME)
    import os
    assert os.path.exists(p)
