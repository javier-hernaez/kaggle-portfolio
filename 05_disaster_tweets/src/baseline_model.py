"""
Modelo Baseline para NLP with Disaster Tweets.
Entrena clasificadores lineales de alta regularización (Logistic Regression,
Calibrated Ridge Classifier y SGD Log-Loss) sobre la representación combinada
(TF-IDF N-gramas de palabras + caracteres + meta-features).

Aplica 5-Fold Stratified K-Fold CV, optimiza el umbral de F1 y genera submissions.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

# Asegurar codificación utf-8 en terminal Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Importar módulos locales
sys.path.append(os.path.dirname(__file__))
from features import TextFeaturePipeline
from evaluate import find_best_threshold

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def train_and_evaluate():
    print("=" * 65)
    print(" CARGANDO DATOS Y CONSTRUYENDO PIPELINE TF-IDF + META-FEATURES")
    print("=" * 65)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train_df['target'].values

    pipeline = TextFeaturePipeline(max_word_features=15000, max_char_features=20000)
    print("[*] Ajustando vectorizadores y procesando textos de Train...")
    X_train = pipeline.fit_transform(train_df)
    print(f"[+] Matriz de entrenamiento: {X_train.shape[0]:,} muestras, {X_train.shape[1]:,} características")

    print("[*] Transformando textos de Test...")
    X_test = pipeline.transform(test_df)
    print(f"[+] Matriz de prueba: {X_test.shape[0]:,} muestras, {X_test.shape[1]:,} características")

    # Validación cruzada 5-Fold Stratified
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lr = np.zeros(len(train_df))
    test_preds_lr = np.zeros(len(test_df))

    oof_preds_ridge = np.zeros(len(train_df))
    test_preds_ridge = np.zeros(len(test_df))

    print("\n" + "=" * 65)
    print(f" ENTRENAMIENTO 5-FOLD STRATIFIED K-FOLD")
    print("=" * 65)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, y_tr = X_train[train_idx], y[train_idx]
        X_val, y_val = X_train[val_idx], y[val_idx]

        # Modelo 1: Logistic Regression con solver rápido liblinear
        clf_lr = LogisticRegression(C=1.2, max_iter=1000, solver='liblinear', random_state=42)
        clf_lr.fit(X_tr, y_tr)
        val_probs_lr = clf_lr.predict_proba(X_val)[:, 1]
        oof_preds_lr[val_idx] = val_probs_lr
        test_preds_lr += clf_lr.predict_proba(X_test)[:, 1] / n_splits

        # Modelo 2: Ridge Classifier con calibración sigmoidea
        base_ridge = RidgeClassifier(alpha=1.5, random_state=42)
        clf_ridge = CalibratedClassifierCV(estimator=base_ridge, method='sigmoid', cv=3)
        clf_ridge.fit(X_tr, y_tr)
        val_probs_ridge = clf_ridge.predict_proba(X_val)[:, 1]
        oof_preds_ridge[val_idx] = val_probs_ridge
        test_preds_ridge += clf_ridge.predict_proba(X_test)[:, 1] / n_splits

        f1_fold_lr = f1_score(y_val, (val_probs_lr >= 0.5).astype(int))
        f1_fold_ridge = f1_score(y_val, (val_probs_ridge >= 0.5).astype(int))
        print(f"Fold {fold}/{n_splits} | F1 LR: {f1_fold_lr:.4f} | F1 Ridge: {f1_fold_ridge:.4f}")

    print("\n" + "=" * 65)
    print(" RESULTADOS INDIVIDUALES (OOF)")
    print("=" * 65)
    print("--- Logistic Regression ---")
    find_best_threshold(y, oof_preds_lr)

    print("\n--- Calibrated Ridge Classifier ---")
    find_best_threshold(y, oof_preds_ridge)

    # Ensamble por promedio ponderado
    print("\n" + "=" * 65)
    print(" ENSAMBLE (50% Logistic Regression + 50% Calibrated Ridge)")
    print("=" * 65)
    oof_ensemble = 0.5 * oof_preds_lr + 0.5 * oof_preds_ridge
    test_ensemble = 0.5 * test_preds_lr + 0.5 * test_preds_ridge

    best_thresh, best_f1, auc = find_best_threshold(y, oof_ensemble)

    # Generar submissions
    # 1. Con umbral estándar 0.50
    # 2. Con umbral optimizado
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_def = pd.DataFrame({'id': test_df['id'], 'target': (test_ensemble >= 0.50).astype(int)})
    sub_opt = pd.DataFrame({'id': test_df['id'], 'target': (test_ensemble >= best_thresh).astype(int)})

    sub_def_path = os.path.join(SUB_DIR, f"submission_ensemble_th0.50_{timestamp}.csv")
    sub_opt_path = os.path.join(SUB_DIR, f"submission_ensemble_opt_th{best_thresh:.3f}_{timestamp}.csv")
    latest_path = os.path.join(BASE_DIR, "submission.csv")

    sub_def.to_csv(sub_def_path, index=False)
    sub_opt.to_csv(sub_opt_path, index=False)
    sub_opt.to_csv(latest_path, index=False)

    print(f"\n[OK] Submissions guardadas:")
    print(f"  -> {sub_def_path} (Positivos: {sub_def['target'].sum()} / {len(sub_def)})")
    print(f"  -> {sub_opt_path} (Positivos: {sub_opt['target'].sum()} / {len(sub_opt)})")
    print(f"  -> {latest_path} (Copia activa para submit)")

    return best_f1, best_thresh

if __name__ == "__main__":
    train_and_evaluate()
