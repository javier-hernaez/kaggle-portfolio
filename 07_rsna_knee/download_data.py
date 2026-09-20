"""
Download and verify competition metadata files for RSNA Knee Abnormality Detection.
"""

import os
import subprocess
from pathlib import Path
import pandas as pd

COMPETITION = "rsna-knee-abnormality-detection"
DATA_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_FILES = [
    "sample_submission.csv",
    "test.csv",
    "test_series.csv",
    "train.csv",
    "train_series.csv",
]


def download_file(filename: str, dest_dir: Path) -> bool:
    print(f"[*] Downloading {filename}...")
    cmd = [
        ".\\.venv\\Scripts\\kaggle.exe",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-f",
        filename,
        "-p",
        str(dest_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error downloading {filename}: {res.stderr}")
        return False
    print(f"[+] Downloaded {filename} successfully.")
    return True


def verify_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for f in REQUIRED_FILES:
        filepath = DATA_DIR / f
        if not filepath.exists():
            missing.append(f)

    if missing:
        print(f"Missing {len(missing)} files: {missing}. Downloading...")
        for f in missing:
            download_file(f, DATA_DIR)
    else:
        print("[+] All metadata files exist locally.")

    print("\nFile status summary:")
    for f in REQUIRED_FILES:
        filepath = DATA_DIR / f
        size_mb = filepath.stat().st_size / (1024 * 1024)
        df = pd.read_csv(filepath, nrows=5)
        print(f" - {f:22s}: {size_mb:6.2f} MB | Columns: {len(df.columns)}")


if __name__ == "__main__":
    verify_files()
