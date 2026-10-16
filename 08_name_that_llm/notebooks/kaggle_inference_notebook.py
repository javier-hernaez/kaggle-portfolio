"""
Standalone Kaggle Inference Notebook for "Name That LLM".
Can be pasted directly into a Kaggle Notebook or submitted as a code script.
Handles path detection, stylometric extraction, n-gram features, multi-engine ensemble,
and exports `submission.parquet`.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

# -----------------------------------------------------------------------------
# 1. Environment & Path Resolution
# -----------------------------------------------------------------------------
KAGGLE_PATHS = [
    "/kaggle/input/name-that-llm",
    "/kaggle/input/competitions/name-that-llm",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data") if "__file__" in locals() else "data",
    "./data",
    "."
]

DATA_DIR = None
for p in KAGGLE_PATHS:
    if os.path.exists(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not locate train.csv in any standard Kaggle input directories.")

print(f"[+] Using dataset from: {DATA_DIR}")

# -----------------------------------------------------------------------------
# 2. Stylometric Feature Extractor
# -----------------------------------------------------------------------------
LLM_MARKERS = [
    "however", "moreover", "furthermore", "crucially", "ultimately", "indeed",
    "thus", "therefore", "notably", "specifically", "essentially", "delve",
    "tapestry", "testament", "beacon", "realm", "intricate", "nuanced",
    "underscores", "underscoring", "fosters", "pivotal", "in essence",
    "at its core", "it is worth noting", "serves as", "a testament to"
]


def extract_stylometrics(texts):
    records = []
    for text in texts:
        if not isinstance(text, str):
            text = ""

        char_len = len(text)
        words = text.split()
        word_len = max(len(words), 1)

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sent_count = max(len(sentences), 1)
        sent_lens = [len(s.split()) for s in sentences] if sentences else [word_len]

        em_dash_count = text.count("—") + text.count("--")
        colon_count = text.count(":")
        semicolon_count = text.count(";")
        comma_count = text.count(",")
        straight_double = text.count('"')
        straight_single = text.count("'")
        curly_double = text.count("“") + text.count("”")
        curly_single = text.count("‘") + text.count("’")
        total_curly = curly_double + curly_single

        newlines = text.count("\n")
        bold_stars = text.count("**")
        contractions = len(re.findall(r"\b\w+(?:'|’)(?:t|s|ve|re|ll|d|m)\b", text, re.IGNORECASE))
        capitals = sum(1 for c in text if c.isupper())
        ttr = len(set(w.lower() for w in words)) / word_len

        factor = 100.0 / word_len
        marker_counts = sum(text.lower().count(m) for m in LLM_MARKERS)

        records.append({
            "char_len": char_len,
            "word_len": word_len,
            "avg_word_len": char_len / word_len,
            "sent_count": sent_count,
            "avg_sent_len": np.mean(sent_lens),
            "newlines": newlines,
            "caps_ratio": capitals / max(char_len, 1),
            "ttr": ttr,
            "em_dash_count": em_dash_count,
            "colon_count": colon_count,
            "semicolon_count": semicolon_count,
            "comma_count": comma_count,
            "straight_quotes": straight_double + straight_single,
            "curly_quotes": total_curly,
            "has_curly": float(total_curly > 0),
            "has_em_dash": float(em_dash_count > 0),
            "em_dash_per_100": em_dash_count * factor,
            "colon_per_100": colon_count * factor,
            "semicolon_per_100": semicolon_count * factor,
            "contractions_per_100": contractions * factor,
            "markers_per_100": marker_counts * factor,
            "bold_stars": bold_stars,
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 3. Main Pipeline
# -----------------------------------------------------------------------------
def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    classes = sorted(train_df["model"].unique())
    label2idx = {name: i for i, name in enumerate(classes)}
    idx2label = {i: name for i, name in enumerate(classes)}
    y = train_df["model"].map(label2idx).values
    groups = train_df["prompt_id"].values

    print("[*] Extracting stylometrics...")
    scaler = StandardScaler()
    sty_train = scaler.fit_transform(extract_stylometrics(train_df["text"]))
    sty_test = scaler.transform(extract_stylometrics(test_df["text"]))

    print("[*] Building N-Gram Text Pipelines...")
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 6),
        analyzer="char",
        max_features=60000,
        sublinear_tf=True,
        min_df=3,
    )

    w_tr = word_vec.fit_transform(train_df["text"])
    w_te = word_vec.transform(test_df["text"])

    c_tr = char_vec.fit_transform(train_df["text"])
    c_te = char_vec.transform(test_df["text"])

    X_train = hstack([csr_matrix(sty_train), w_tr, c_tr]).tocsr()
    X_test = hstack([csr_matrix(sty_test), w_te, c_te]).tocsr()

    print(f"Total features: {X_train.shape[1]:,}")

    # 5-Fold Stratified Group KFold Cross Validation
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_probs = np.zeros((len(test_df), len(classes)))
    oof_probs = np.zeros((len(train_df), len(classes)))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        X_fold_tr, y_fold_tr = X_train[train_idx], y[train_idx]
        X_fold_va, y_fold_va = X_train[val_idx], y[val_idx]

        # Engine 1: Logistic Regression
        clf_lr = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", n_jobs=-1, random_state=42)
        clf_lr.fit(X_fold_tr, y_fold_tr)
        val_probs_lr = clf_lr.predict_proba(X_fold_va)
        test_probs_lr = clf_lr.predict_proba(X_test)

        # Engine 2: Ridge Classifier
        clf_ridge = RidgeClassifier(alpha=1.0, random_state=42)
        clf_ridge.fit(X_fold_tr, y_fold_tr)
        val_probs_ridge = softmax(clf_ridge.decision_function(X_fold_va), axis=1)
        test_probs_ridge = softmax(clf_ridge.decision_function(X_test), axis=1)

        # Ensemble
        fold_val_probs = 0.60 * val_probs_lr + 0.40 * val_probs_ridge
        oof_probs[val_idx] = fold_val_probs
        fold_test_probs = 0.60 * test_probs_lr + 0.40 * test_probs_ridge
        test_probs += fold_test_probs / n_splits

        fold_acc = balanced_accuracy_score(y_fold_va, np.argmax(fold_val_probs, axis=1))
        print(f"Fold {fold + 1} Balanced Accuracy: {fold_acc:.4f}")

    total_acc = balanced_accuracy_score(y, np.argmax(oof_probs, axis=1))
    print(f"\n[🏆] Total Out-Of-Fold Balanced Accuracy: {total_acc:.4f}")

    # Generate submission.parquet
    test_pred_idx = np.argmax(test_probs, axis=1)
    submission = pd.DataFrame({
        "id": test_df["id"],
        "model": [idx2label[i] for i in test_pred_idx],
    })

    submission.to_parquet("submission.parquet", index=False)
    print("[✔] Successfully saved submission.parquet!")
    print(submission.head())


if __name__ == "__main__":
    main()
