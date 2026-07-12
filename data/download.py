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
