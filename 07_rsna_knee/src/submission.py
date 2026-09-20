"""
Submission generation and validation pipeline for RSNA Knee Abnormality Detection.
Verifies:
  1. Exact matching column names and order with sample_submission.csv.
  2. StudyInstanceUID integrity against test.csv.
  3. Prediction bounds [0.0, 1.0] and absence of NaNs / infinities.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def validate_submission(sub_df: pd.DataFrame) -> bool:
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    expected_cols = sample_sub.columns.tolist()
    if sub_df.columns.tolist() != expected_cols:
        raise ValueError(f"Column mismatch! Expected: {expected_cols}, got: {sub_df.columns.tolist()}")

    if len(sub_df) != len(sample_sub):
        raise ValueError(f"Row count mismatch! Expected: {len(sample_sub)}, got: {len(sub_df)}")

    if (sub_df["StudyInstanceUID"].values != test_df["StudyInstanceUID"].values).any():
        raise ValueError("StudyInstanceUIDs do not match test.csv exactly in order!")

    # Check for nulls/infinities
    if sub_df.isna().any().any():
        null_cols = sub_df.columns[sub_df.isna().any()].tolist()
        raise ValueError(f"Found NaN values in submission columns: {null_cols}")

    # Check bounds
    numeric_data = sub_df[TARGET_COLUMNS].values
    if (numeric_data < 0.0).any() or (numeric_data > 1.0).any():
        raise ValueError("Probabilities out of bounds [0.0, 1.0]!")

    print("[+] Submission passed all validation checks successfully!")
    return True


def save_and_verify_submission(preds_df: pd.DataFrame, output_name: str = "baseline_submission.csv") -> Path:
    out_path = OUTPUT_DIR / output_name

    # Order columns
    cols = ["StudyInstanceUID"] + TARGET_COLUMNS
    final_sub = preds_df[cols].copy()

    validate_submission(final_sub)

    final_sub.to_csv(out_path, index=False)
    print(f"[+] Saved verified submission to: {out_path}")
    print(f"    Rows: {len(final_sub)}, Cols: {len(final_sub.columns)}")
    print(final_sub.head(3))
    return out_path


if __name__ == "__main__":
    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from .model import MultiLabelKneePredictor
    except ImportError:
        from model import MultiLabelKneePredictor

    print("=" * 65)
    print("      GENERATING AND VALIDATING BASELINE SUBMISSION             ")
    print("=" * 65)
    predictor = MultiLabelKneePredictor(use_weak_supervision=True)
    train_merged, test_merged = predictor.prepare_data()
    test_preds = predictor.fit_predict(train_merged, test_merged)
    save_and_verify_submission(test_preds, output_name="baseline_submission.csv")
