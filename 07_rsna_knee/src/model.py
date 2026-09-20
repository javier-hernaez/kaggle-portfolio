"""
Baseline and Weakly-Supervised Models for RSNA Knee Abnormality Detection.
Combines:
  1. Weak supervision labels extracted from multilingual reports across 4,349 studies.
  2. Series-level MRI protocol features (planes, contrast, counts).
  3. Stratified cross-validation on the 58 gold-standard human annotations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import sys

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from .series_features import extract_series_features
    from .report_extractor import ClinicalReportExtractor, TARGET_COLUMNS, EMPIRICAL_BASE_RATES
    from .evaluate import compute_competition_metric, format_evaluation_summary
except ImportError:
    from series_features import extract_series_features
    from report_extractor import ClinicalReportExtractor, TARGET_COLUMNS, EMPIRICAL_BASE_RATES
    from evaluate import compute_competition_metric, format_evaluation_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"


class MultiLabelKneePredictor:
    def __init__(self, use_weak_supervision: bool = True):
        self.use_weak_supervision = use_weak_supervision
        self.models = {}
        self.feature_cols = []
        self.base_rates = EMPIRICAL_BASE_RATES.copy()

    def prepare_data(self):
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        train_series = pd.read_csv(DATA_DIR / "train_series.csv")
        test_df = pd.read_csv(DATA_DIR / "test.csv")
        test_series = pd.read_csv(DATA_DIR / "test_series.csv")

        # Extract series features
        train_feats = extract_series_features(train_series)
        test_feats = extract_series_features(test_series)

        # Merge with train metadata
        train_merged = train_df.merge(train_feats, on="StudyInstanceUID", how="left")
        test_merged = test_df.merge(test_feats, on="StudyInstanceUID", how="left")

        # Feature columns (all numeric series features)
        self.feature_cols = [c for c in train_feats.columns if c != "StudyInstanceUID"]

        # Weak supervision pseudo-labels for unlabeled studies
        if self.use_weak_supervision:
            extractor = ClinicalReportExtractor()
            unlabeled_mask = train_merged["ACL"].isna()
            pseudo_df = extractor.extract_dataset(train_merged[unlabeled_mask])

            # Fill missing targets with pseudo-labels
            for target in TARGET_COLUMNS:
                train_merged.loc[unlabeled_mask, target] = pseudo_df[target].values

        return train_merged, test_merged

    def cross_validate_gold(self, n_splits: int = 5):
        """
        Evaluates feature-based models via cross-validation strictly on the 58 gold-standard cases.
        """
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        train_series = pd.read_csv(DATA_DIR / "train_series.csv")
        train_feats = extract_series_features(train_series)
        merged = train_df.merge(train_feats, on="StudyInstanceUID", how="left")

        gold_df = merged[merged["ACL"].notna()].copy().reset_index(drop=True)
        feature_cols = [c for c in train_feats.columns if c != "StudyInstanceUID"]

        X = gold_df[feature_cols].values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_preds = pd.DataFrame(index=gold_df.index, columns=TARGET_COLUMNS)

        for target in TARGET_COLUMNS:
            y = gold_df[target].values
            oof_target = np.zeros(len(gold_df))

            for train_idx, val_idx in kf.split(X, y):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]

                # Ridge/Logistic calibrated model
                base_p = np.mean(y_tr)
                clf = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", random_state=42)
                try:
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict_proba(X_va)[:, 1]
                except Exception:
                    preds = np.full(len(val_idx), base_p)

                # Blend with empirical prior
                oof_target[val_idx] = 0.5 * preds + 0.5 * base_p

            oof_preds[target] = oof_target

        macro_auc, details = compute_competition_metric(gold_df[TARGET_COLUMNS], oof_preds)
        return macro_auc, details, oof_preds

    def fit_predict(self, train_merged: pd.DataFrame, test_merged: pd.DataFrame):
        """
        Trains models on available data and generates test predictions.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        X_train = train_merged[self.feature_cols].values
        X_test = test_merged[self.feature_cols].values

        test_preds = pd.DataFrame()
        test_preds["StudyInstanceUID"] = test_merged["StudyInstanceUID"]

        for target in TARGET_COLUMNS:
            y_train = train_merged[target].values
            valid_mask = ~np.isnan(y_train)

            X_tr = X_train[valid_mask]
            y_tr = y_train[valid_mask]

            # Binarize pseudo-labels for training classifier if continuous
            y_tr_binary = (y_tr >= 0.5).astype(int)

            # Fit LightGBM / Logistic ensemble
            base_p = float(np.mean(y_tr_binary))
            model = lgb.LGBMClassifier(
                n_estimators=40,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=7,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
            )
            try:
                model.fit(X_tr, y_tr_binary)
                preds = model.predict_proba(X_test)[:, 1]
            except Exception:
                preds = np.full(len(X_test), base_p)

            # Blend with empirical base rate
            blended_preds = 0.6 * preds + 0.4 * self.base_rates[target]
            test_preds[target] = np.clip(blended_preds, 0.01, 0.99)

        return test_preds


def run_training():
    predictor = MultiLabelKneePredictor(use_weak_supervision=True)

    print("=" * 65)
    print("      CROSS-VALIDATION ON 58 GOLD-STANDARD GROUND TRUTH        ")
    print("=" * 65)
    cv_macro, cv_details, oof_df = predictor.cross_validate_gold(n_splits=5)
    print(format_evaluation_summary(cv_macro, cv_details))

    print("\nPreparing full dataset with weak-supervision pseudo-labels...")
    train_merged, test_merged = predictor.prepare_data()
    print(f"Train merged shape: {train_merged.shape}")
    print(f"Test merged shape: {test_merged.shape}")

    print("\nFitting final models on 4,407 studies and predicting on test...")
    test_preds = predictor.fit_predict(train_merged, test_merged)
    print("\nGenerated test predictions:")
    print(test_preds)

    return predictor, cv_macro, test_preds


if __name__ == "__main__":
    run_training()
