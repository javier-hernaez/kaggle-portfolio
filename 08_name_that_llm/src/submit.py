"""
Submission Generator and Validator for Name That LLM.
Produces a compliant `submission.parquet` file and validates schema against `sample_submission.parquet`.
"""

import os
import sys
import pickle
import datetime
import pandas as pd
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def generate_submission():
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_path = os.path.join(DATA_DIR, "sample_submission.parquet")
    artifact_path = os.path.join(MODELS_DIR, "oof_results.pkl")

    print("=" * 80)
    print(" 📦 GENERATING KAGGLE SUBMISSION: Name That LLM")
    print("=" * 80)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    test_df = pd.read_csv(test_path)

    # Load trained model results
    if not os.path.exists(artifact_path):
        print("[*] Model artifacts not found. Training model now...")
        from model import train_and_evaluate
        artifacts = train_and_evaluate()
    else:
        with open(artifact_path, "rb") as f:
            artifacts = pickle.load(f)

    test_pred_labels = artifacts["test_pred_labels"]
    classes = artifacts["classes"]

    # Verify lengths
    if len(test_pred_labels) != len(test_df):
        raise ValueError(
            f"Length mismatch: {len(test_pred_labels)} predictions vs {len(test_df)} test rows."
        )

    # Construct DataFrame
    sub_df = pd.DataFrame({
        "id": test_df["id"],
        "model": test_pred_labels,
    })

    # File paths
    target_parquet = os.path.join(BASE_DIR, "submission.parquet")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_parquet = os.path.join(SUBMISSIONS_DIR, f"submission_{timestamp}.parquet")

    # Save to Parquet
    sub_df.to_parquet(target_parquet, index=False)
    sub_df.to_parquet(timestamped_parquet, index=False)
    print(f"[✔] Submission file saved to: {target_parquet}")
    print(f"[✔] Archival copy saved to: {timestamped_parquet}")

    # =========================================================================
    # Validation against Kaggle specifications
    # =========================================================================
    print("\n" + "-" * 80)
    print(" 🔍 VALIDATING SUBMISSION CONSTRAINTS")
    print("-" * 80)

    # 1. Row count
    print(f"[*] Total rows: {len(sub_df):,} (Expected: {len(test_df):,}) -> {'PASS' if len(sub_df) == len(test_df) else 'FAIL'}")

    # 2. Columns
    expected_cols = ["id", "model"]
    cols_match = list(sub_df.columns) == expected_cols
    print(f"[*] Columns: {list(sub_df.columns)} (Expected: {expected_cols}) -> {'PASS' if cols_match else 'FAIL'}")

    # 3. Null values
    null_count = sub_df.isnull().sum().sum()
    print(f"[*] Null values count: {null_count} -> {'PASS' if null_count == 0 else 'FAIL'}")

    # 4. Valid class labels
    invalid_labels = set(sub_df["model"]) - set(classes)
    print(f"[*] Invalid class labels: {len(invalid_labels)} -> {'PASS' if len(invalid_labels) == 0 else 'FAIL'}")
    if invalid_labels:
        print(f"    [!] Unknown labels found: {invalid_labels}")

    # 5. Schema check against sample_submission.parquet if present
    if os.path.exists(sample_path):
        sample_df = pd.read_parquet(sample_path)
        print(f"[*] Comparing schema against sample_submission.parquet:")
        print(f"    Sample columns: {list(sample_df.columns)}")
        print(f"    Sample shape:   {sample_df.shape}")
        assert list(sub_df.columns) == list(sample_df.columns), "Columns do not match sample submission!"

    print("-" * 80)
    print(" Preview of predictions:")
    print(sub_df.head(10))
    print("=" * 80)
    print(" To submit to Kaggle CLI, run:")
    print(f' kaggle competitions submit -c name-that-llm -f "{target_parquet}" -m "Baseline Stylometric + Multi-Engine Ensemble"')
    print("=" * 80)


if __name__ == "__main__":
    generate_submission()
