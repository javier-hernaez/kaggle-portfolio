"""
Download and verify competition datasets for NFL Big Data Bowl 2027.
Theme: Combine to NFL Performance
"""

import os
from pathlib import Path
import pandas as pd

COMPETITION = "nfl-big-data-bowl-2027"
DATA_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_FILES = [
    "players.csv",
    "combine_results.csv",
    "combine_tracking.csv",
    "player_career_successes.csv",
    "games.csv",
    "player_play.csv",
]


def download_file(filename: str, dest_dir: Path) -> bool:
    print(f"[*] Downloading {filename} via Kaggle API...")
    try:
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
        api.competition_download_file(
            COMPETITION,
            f"{COMPETITION}/{filename}",
            path=str(dest_dir),
            force=False
        )
        print(f"[+] Downloaded {filename} successfully.")
        return True
    except Exception as e:
        print(f"[-] Error downloading {filename}: {e}")
        return False


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
        print("[+] All essential files exist locally.")

    print("\nFile status summary:")
    for f in REQUIRED_FILES:
        filepath = DATA_DIR / f
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  - {f}: {size_mb:.2f} MB")
        else:
            print(f"  - {f}: MISSING")


if __name__ == "__main__":
    verify_files()
