"""
Feature engineering from MRI series metadata (train_series.csv, test_series.csv).
Transforms sequence-level attributes (planes, fat suppression, fluid sensitivity)
into study-level tabular feature representations available at both train and test time.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_series_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates MRI series metadata per StudyInstanceUID into clinical protocol features.
    """
    df = series_df.copy()

    # Pre-calculate interaction flags
    df["is_sagittal"] = (df["Anatomical_Plane"] == "Sagittal").astype(int)
    df["is_coronal"] = (df["Anatomical_Plane"] == "Coronal").astype(int)
    df["is_axial"] = (df["Anatomical_Plane"] == "Axial").astype(int)

    df["fluid_fs"] = ((df["Fluid_Sensitive"] == 1) & (df["Fat_Suppression"] == 1)).astype(int)
    df["sagittal_fluid_fs"] = (df["is_sagittal"] & df["fluid_fs"]).astype(int)
    df["coronal_fluid_fs"] = (df["is_coronal"] & df["fluid_fs"]).astype(int)
    df["axial_fluid_fs"] = (df["is_axial"] & df["fluid_fs"]).astype(int)

    df["sagittal_no_fs"] = (df["is_sagittal"] & (df["Fat_Suppression"] == 0)).astype(int)
    df["coronal_no_fs"] = (df["is_coronal"] & (df["Fat_Suppression"] == 0)).astype(int)
    df["axial_no_fs"] = (df["is_axial"] & (df["Fat_Suppression"] == 0)).astype(int)

    # Group by study
    grouped = df.groupby("StudyInstanceUID")

    feat_df = pd.DataFrame(index=grouped.indices.keys())
    feat_df.index.name = "StudyInstanceUID"

    feat_df["n_series_total"] = grouped.size()
    feat_df["n_sagittal"] = grouped["is_sagittal"].sum()
    feat_df["n_coronal"] = grouped["is_coronal"].sum()
    feat_df["n_axial"] = grouped["is_axial"].sum()

    # Proportions
    feat_df["pct_sagittal"] = feat_df["n_sagittal"] / feat_df["n_series_total"]
    feat_df["pct_coronal"] = feat_df["n_coronal"] / feat_df["n_series_total"]
    feat_df["pct_axial"] = feat_df["n_axial"] / feat_df["n_series_total"]

    # Binary flags for plane presence
    feat_df["has_all_3_planes"] = (
        (feat_df["n_sagittal"] > 0) & (feat_df["n_coronal"] > 0) & (feat_df["n_axial"] > 0)
    ).astype(int)

    # Contrast & suppression aggregations
    feat_df["n_fluid_sensitive"] = grouped["Fluid_Sensitive"].sum()
    feat_df["pct_fluid_sensitive"] = feat_df["n_fluid_sensitive"] / feat_df["n_series_total"]
    feat_df["n_fat_suppression"] = grouped["Fat_Suppression"].sum()
    feat_df["pct_fat_suppression"] = feat_df["n_fat_suppression"] / feat_df["n_series_total"]

    # Protocol combinations
    feat_df["n_sagittal_fluid_fs"] = grouped["sagittal_fluid_fs"].sum()
    feat_df["n_coronal_fluid_fs"] = grouped["coronal_fluid_fs"].sum()
    feat_df["n_axial_fluid_fs"] = grouped["axial_fluid_fs"].sum()
    feat_df["n_sagittal_no_fs"] = grouped["sagittal_no_fs"].sum()
    feat_df["n_coronal_no_fs"] = grouped["coronal_no_fs"].sum()
    feat_df["n_axial_no_fs"] = grouped["axial_no_fs"].sum()

    return feat_df.reset_index()


def load_all_series_features():
    train_series = pd.read_csv(DATA_DIR / "train_series.csv")
    test_series = pd.read_csv(DATA_DIR / "test_series.csv")

    train_feat = extract_series_features(train_series)
    test_feat = extract_series_features(test_series)

    print(f"Extracted train series features: {train_feat.shape}")
    print(f"Extracted test series features: {test_feat.shape}")
    return train_feat, test_feat


if __name__ == "__main__":
    train_feat, test_feat = load_all_series_features()
    print("\nSample train features:")
    print(train_feat.head(3))
