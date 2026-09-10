"""
Electric Vehicle Purchases - Advanced Ensemble Model
Playground Series Season 6 Episode 9

Pipeline:
- Stratified 5-Fold Cross-Validation
- Out-Of-Fold (OOF) ROC-AUC Tracking
- Models: LightGBM + CatBoost + XGBoost
- Blended Ensembling & Test Fold Averaging
- Produces valid competition submission with Will_Buy_EV probabilities
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from features import build_features
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def train_and_evaluate():
    start_time = time.time()
    print("=" * 75)
    print(" PLAYGROUND SERIES S6E9: PREDICTING EV PURCHASES")
    print(" ENSEMBLE PIPELINE (LIGHTGBM + CATBOOST + XGBOOST)")
    print("=" * 75)

    print("\n[*] Cargando datos de entrenamiento y prueba...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_ids = test_raw['id']

    print(f"    - Filas Train: {len(train_raw):,}")
    print(f"    - Filas Test : {len(test_raw):,}")

    print("\n[*] Generando variables (Feature Engineering & DGP Probit Link)...")
    X, y, feature_names = build_features(train_raw, is_train=True)
    X_test, _, _ = build_features(test_raw, is_train=False)

    print(f"    - Total características construidas: {len(feature_names)}")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Contenedores para predicciones OOF y Test
    oof_lgb = np.zeros(len(X))
    oof_cb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))

    print(f"\n[*] Iniciando validación cruzada estratificada ({n_splits} Folds)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()
        print(f"\n" + "-" * 50)
        print(f" Fold {fold} / {n_splits}")
        print("-" * 50)

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 1. LightGBM
        print("  [>] Entrenando LightGBM...")
        model_lgb = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits
        auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
        print(f"      Fold {fold} LightGBM AUC: {auc_lgb:.6f}")

        # 2. CatBoost
        print("  [>] Entrenando CatBoost...")
        model_cb = CatBoostClassifier(
            iterations=400,
            learning_rate=0.08,
            depth=6,
            eval_metric='AUC',
            random_seed=42 + fold,
            thread_count=-1,
            verbose=0
        )
        model_cb.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_val),
            early_stopping_rounds=30,
            verbose=False
        )
        oof_cb[val_idx] = model_cb.predict_proba(X_va)[:, 1]
        test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / n_splits
        auc_cb = roc_auc_score(y_val, oof_cb[val_idx])
        print(f"      Fold {fold} CatBoost AUC: {auc_cb:.6f}")

        # 3. XGBoost
        print("  [>] Entrenando XGBoost...")
        model_xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=30
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            verbose=False
        )
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
        test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits
        auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
        print(f"      Fold {fold} XGBoost AUC : {auc_xgb:.6f}")

        # Ensamble local de este fold
        fold_blend = (oof_lgb[val_idx] * 0.35 + oof_cb[val_idx] * 0.35 + oof_xgb[val_idx] * 0.30)
        auc_blend = roc_auc_score(y_val, fold_blend)
        print(f"      --> Fold {fold} Blend AUC: {auc_blend:.6f} (Tiempo fold: {time.time() - fold_start:.1f}s)")

    # Métricas Globales OOF
    print("\n" + "=" * 75)
    print(" RESULTADOS GLOBALES DE VALIDACIÓN CRUZADA (OOF ROC-AUC)")
    print("=" * 75)
    total_auc_lgb = roc_auc_score(y, oof_lgb)
    total_auc_cb = roc_auc_score(y, oof_cb)
    total_auc_xgb = roc_auc_score(y, oof_xgb)

    # Búsqueda de pesos óptimos para el ensamble
    best_weights = (0.35, 0.35, 0.30)
    best_ensemble_auc = 0
    for w_lgb in np.linspace(0.1, 0.6, 6):
        for w_cb in np.linspace(0.1, 0.6, 6):
            w_xgb = 1.0 - w_lgb - w_cb
            if w_xgb < 0:
                continue
            curr_oof = w_lgb * oof_lgb + w_cb * oof_cb + w_xgb * oof_xgb
            curr_auc = roc_auc_score(y, curr_oof)
            if curr_auc > best_ensemble_auc:
                best_ensemble_auc = curr_auc
                best_weights = (w_lgb, w_cb, w_xgb)

    w1, w2, w3 = best_weights
    oof_ensemble = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    final_test_preds = w1 * test_preds_lgb + w2 * test_preds_cb + w3 * test_preds_xgb

    print(f"  LightGBM OOF ROC-AUC : {total_auc_lgb:.6f}")
    print(f"  CatBoost OOF ROC-AUC : {total_auc_cb:.6f}")
    print(f"  XGBoost OOF ROC-AUC  : {total_auc_xgb:.6f}")
    print(f"  Pesos Óptimos        : LGB={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f}")
    print(f"  [*] ENSAMBLE OOF ROC-AUC: {best_ensemble_auc:.6f}")

    # Guardar predicciones
    sub_df = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': np.clip(final_test_preds, 0.0, 1.0)
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    versioned_sub_path = os.path.join(SUB_DIR, f"submission_blend_{timestamp}_auc_{best_ensemble_auc:.5f}.csv")
    final_sub_path = os.path.join(os.path.dirname(__file__), "..", "submission.csv")

    sub_df.to_csv(versioned_sub_path, index=False)
    sub_df.to_csv(final_sub_path, index=False)

    print(f"\n[+] Archivos de sumisión guardados exitosamente:")
    print(f"    - Historial: {versioned_sub_path}")
    print(f"    - Principal: {final_sub_path}")
    print(f"    - Total filas: {len(sub_df):,}")
    print(f"    - Rango probabilidades: [{sub_df['Will_Buy_EV'].min():.4f}, {sub_df['Will_Buy_EV'].max():.4f}]")
    print(f"    - Media de compra predicha: {sub_df['Will_Buy_EV'].mean():.2%}")
    print(f"\nTiempo total de ejecución: {(time.time() - start_time) / 60:.2f} minutos.")

    return best_ensemble_auc

if __name__ == "__main__":
    train_and_evaluate()
