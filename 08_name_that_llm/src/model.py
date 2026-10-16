"""
Modeling and Validation Pipeline for Name That LLM.
Features:
- StratifiedGroupKFold on prompt_id to prevent topic/prompt leakage.
- Multi-engine model ensemble (Logistic Regression + Calibrated Ridge + LightGBM on Stylometrics/SVD).
- Evaluates against the official Kaggle metric: Balanced Accuracy Score.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path for local execution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import extract_all_features, StylometricExtractor, build_text_pipelines
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_and_evaluate(
    n_splits: int = 5,
    random_state: int = 42,
    max_features_word: int = 20000,
    max_features_char: int = 50000,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    print("=" * 80)
    print(" 🚀 NAME THAT LLM - 5-FOLD STRATIFIED GROUP CV TRAINING ENGINE")
    print("=" * 80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Encode target labels
    classes = sorted(train_df["model"].unique())
    label2idx = {name: i for i, name in enumerate(classes)}
    idx2label = {i: name for i, name in enumerate(classes)}
    train_df["target"] = train_df["model"].map(label2idx)

    print(f"[*] Total training passages: {len(train_df)}")
    print(f"[*] Total test passages: {len(test_df)}")
    print(f"[*] Classes ({len(classes)}): {classes[:4]} ... {classes[-2:]}")

    # Groups & Cross-Validation Strategy
    groups = train_df["prompt_id"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds_lr = np.zeros((len(train_df), len(classes)))
    oof_preds_ridge = np.zeros((len(train_df), len(classes)))
    oof_preds_lgb = np.zeros((len(train_df), len(classes)))
    oof_preds_ensemble = np.zeros((len(train_df), len(classes)))

    test_preds_lr = np.zeros((len(test_df), len(classes)))
    test_preds_ridge = np.zeros((len(test_df), len(classes)))
    test_preds_lgb = np.zeros((len(test_df), len(classes)))

    # Compute dense stylometrics globally once (unsupervised, no leakage)
    print("[*] Computing dense stylometrics...")
    sty_extractor = StylometricExtractor(scale=True)
    dense_sty_train = sty_extractor.fit_transform(train_df["text"])
    dense_sty_test = sty_extractor.transform(test_df["text"])

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df["target"], groups=groups)):
        print(f"\n" + "-" * 70)
        print(f"  FOLD {fold + 1}/{n_splits} | Train: {len(train_idx)} | Val: {len(val_idx)} (Unique Val Prompts: {len(set(groups[val_idx]))})")
        print("-" * 70)

        # Build TF-IDF strictly fitted on train fold to avoid topic leakage
        word_vec, char_vec = build_text_pipelines(
            max_features_word=max_features_word,
            max_features_char=max_features_char,
        )
        w_tr = word_vec.fit_transform(train_df["text"].iloc[train_idx])
        w_va = word_vec.transform(train_df["text"].iloc[val_idx])
        w_te = word_vec.transform(test_df["text"])

        c_tr = char_vec.fit_transform(train_df["text"].iloc[train_idx])
        c_va = char_vec.transform(train_df["text"].iloc[val_idx])
        c_te = char_vec.transform(test_df["text"])

        # Stack features
        sty_tr = csr_matrix(dense_sty_train[train_idx])
        sty_va = csr_matrix(dense_sty_train[val_idx])
        sty_te = csr_matrix(dense_sty_test)

        X_tr = hstack([sty_tr, w_tr, c_tr]).tocsr()
        X_va = hstack([sty_va, w_va, c_va]).tocsr()
        X_te = hstack([sty_te, w_te, c_te]).tocsr()

        y_tr = train_df["target"].iloc[train_idx].values
        y_va = train_df["target"].iloc[val_idx].values

        # -------------------------------------------------------------
        # Engine 1: Regularized Logistic Regression (Multinomial / L2)
        # -------------------------------------------------------------
        print("    -> Training Engine 1: Logistic Regression...")
        clf_lr = LogisticRegression(
            C=1.5,
            max_iter=300,
            solver="lbfgs",
            n_jobs=-1,
            random_state=random_state,
        )
        clf_lr.fit(X_tr, y_tr)
        val_probs_lr = clf_lr.predict_proba(X_va)
        oof_preds_lr[val_idx] = val_probs_lr
        test_preds_lr += clf_lr.predict_proba(X_te) / n_splits

        score_lr = balanced_accuracy_score(y_va, np.argmax(val_probs_lr, axis=1))

        # -------------------------------------------------------------
        # Engine 2: Ridge Classifier with Softmax Probability Scaling
        # -------------------------------------------------------------
        print("    -> Training Engine 2: Ridge Classifier...")
        clf_ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
        clf_ridge.fit(X_tr, y_tr)
        val_decision_ridge = clf_ridge.decision_function(X_va)
        val_probs_ridge = softmax(val_decision_ridge, axis=1)
        oof_preds_ridge[val_idx] = val_probs_ridge
        test_preds_ridge += softmax(clf_ridge.decision_function(X_te), axis=1) / n_splits

        score_ridge = balanced_accuracy_score(y_va, np.argmax(val_probs_ridge, axis=1))

        # -------------------------------------------------------------
        # Engine 3: LightGBM on Stylometrics + SVD Components
        # -------------------------------------------------------------
        print("    -> Training Engine 3: LightGBM (Stylometrics + SVD)...")
        svd = TruncatedSVD(n_components=64, random_state=random_state)
        svd_tr = svd.fit_transform(hstack([w_tr, c_tr]))
        svd_va = svd.transform(hstack([w_va, c_va]))
        svd_te = svd.transform(hstack([w_te, c_te]))

        lgb_tr = np.hstack([dense_sty_train[train_idx], svd_tr])
        lgb_va = np.hstack([dense_sty_train[val_idx], svd_va])
        lgb_te = np.hstack([dense_sty_test, svd_te])

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=len(classes),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        clf_lgb.fit(lgb_tr, y_tr)
        val_probs_lgb = clf_lgb.predict_proba(lgb_va)
        oof_preds_lgb[val_idx] = val_probs_lgb
        test_preds_lgb += clf_lgb.predict_proba(lgb_te) / n_splits

        score_lgb = balanced_accuracy_score(y_va, np.argmax(val_probs_lgb, axis=1))

        # -------------------------------------------------------------
        # Fold Ensemble Fusion (Weighted Average)
        # -------------------------------------------------------------
        val_probs_ens = 0.50 * val_probs_lr + 0.35 * val_probs_ridge + 0.15 * val_probs_lgb
        oof_preds_ensemble[val_idx] = val_probs_ens
        fold_acc = balanced_accuracy_score(y_va, np.argmax(val_probs_ens, axis=1))
        fold_scores.append(fold_acc)

        print(
            f"  [Fold {fold + 1} Results] LogReg: {score_lr:.4f} | Ridge: {score_ridge:.4f} | "
            f"LGB: {score_lgb:.4f} | Ensemble: {fold_acc:.4f}"
        )

    # -----------------------------------------------------------------
    # Global OOF Evaluation
    # -----------------------------------------------------------------
    y_true = train_df["target"].values
    oof_pred_labels = np.argmax(oof_preds_ensemble, axis=1)

    overall_lr = balanced_accuracy_score(y_true, np.argmax(oof_preds_lr, axis=1))
    overall_ridge = balanced_accuracy_score(y_true, np.argmax(oof_preds_ridge, axis=1))
    overall_lgb = balanced_accuracy_score(y_true, np.argmax(oof_preds_lgb, axis=1))
    overall_ensemble = balanced_accuracy_score(y_true, oof_pred_labels)

    print("\n" + "=" * 80)
    print(" 🏆 FINAL OUT-OF-FOLD CROSS-VALIDATION SUMMARY (BALANCED ACCURACY)")
    print("=" * 80)
    print(f" Engine 1 (Logistic Regression):   {overall_lr:.5f}")
    print(f" Engine 2 (Calibrated Ridge):      {overall_ridge:.5f}")
    print(f" Engine 3 (LightGBM Stacking):     {overall_lgb:.5f}")
    print(f" Multi-Engine Ensemble (Blended):  {overall_ensemble:.5f}")
    print(f" Mean Fold Score:                  {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    print("=" * 80)

    # Classification Report
    print("\nDetailed Per-Class Classification Report (OOF Ensemble):")
    report = classification_report(
        y_true,
        oof_pred_labels,
        target_names=classes,
        digits=4,
    )
    print(report)

    # Save artifacts & test predictions
    test_ensemble_probs = 0.50 * test_preds_lr + 0.35 * test_preds_ridge + 0.15 * test_preds_lgb
    test_pred_indices = np.argmax(test_ensemble_probs, axis=1)
    test_pred_labels = [idx2label[idx] for idx in test_pred_indices]

    artifacts = {
        "classes": classes,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "oof_preds": oof_preds_ensemble,
        "test_probs": test_ensemble_probs,
        "test_pred_labels": test_pred_labels,
        "overall_score": overall_ensemble,
    }

    artifact_path = os.path.join(MODELS_DIR, "oof_results.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[✔] Cross-validation artifacts saved to {artifact_path}")

    return artifacts


if __name__ == "__main__":
    train_and_evaluate()
