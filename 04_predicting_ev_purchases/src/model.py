"""
Grandmaster Super-Ensemble Model (v3 - Top 30 World Edition)
Playground Series Season 6 Episode 9: Predicting Electric Vehicle Purchases

Key Innovations:
1. 118 Features: Original Dataset Target Means, Digit Decomposition (10^-4 to 10^3),
   Regime Clifs/Spikes, Probit DGP Prior, Simpson's Paradox Charging Interactions.
2. In-Fold Triple Target Encoding (smooth='auto', smooth=10.0, smooth=100.0) without data leakage.
3. High-Resolution Tree Models (max_bin=1024, num_leaves=32, colsample_bytree=0.3).
4. Dual Model Architectures: High-Capacity LightGBM + Deep-Hist XGBoost.
5. Optimal Rank Averaging across 10 models (5 folds x 2 models).
"""

import os
import gc
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

sys.path.append(os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from features import build_grandmaster_features
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def train_grandmaster_ensemble():
    start_time = time.time()
    print("=" * 80)
    print(" PLAYGROUND SERIES S6E9: ASALTO AL TOP 30 MUNDIAL (EDICIÓN V3)")
    print(" 118 FEATURES + TRIPLE TARGET ENCODING + HISTOGRAMAS MAX_BIN=1024")
    print("=" * 80)

    # 1. Cargar matriz de características Grandmaster
    X_train, y_train, X_test, test_ids, target_encode_cols = build_grandmaster_features()
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_lgb_rank = np.zeros(len(X_train))
    oof_xgb_rank = np.zeros(len(X_train))

    test_preds_lgb_rank = np.zeros(len(X_test))
    test_preds_xgb_rank = np.zeros(len(X_test))

    print(f"\n[*] Iniciando validación cruzada estratificada ({n_splits} Folds)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        fold_start = time.time()
        print(f"\n" + "-" * 60)
        print(f" Fold {fold} / {n_splits}")
        print("-" * 60)

        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y_train.iloc[train_idx].values
        X_va = X_train.iloc[val_idx].copy()
        y_val = y_train.iloc[val_idx].values
        X_te = X_test.copy()

        # In-Fold Triple Target Encoding
        print(f"  [*] Aplicando Triple Target Encoding en {len(target_encode_cols)} variables...")
        te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
        te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
        te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)

        enc_auto_tr = te_auto.fit_transform(X_tr[target_encode_cols], y_tr)
        enc_auto_va = te_auto.transform(X_va[target_encode_cols])
        enc_auto_te = te_auto.transform(X_te[target_encode_cols])

        enc_10_tr = te_10.fit_transform(X_tr[target_encode_cols], y_tr)
        enc_10_va = te_10.transform(X_va[target_encode_cols])
        enc_10_te = te_10.transform(X_te[target_encode_cols])

        enc_100_tr = te_100.fit_transform(X_tr[target_encode_cols], y_tr)
        enc_100_va = te_100.transform(X_va[target_encode_cols])
        enc_100_te = te_100.transform(X_te[target_encode_cols])

        for i, col in enumerate(target_encode_cols):
            X_tr[f"{col}_TE_auto"] = enc_auto_tr[:, i].astype('float32')
            X_va[f"{col}_TE_auto"] = enc_auto_va[:, i].astype('float32')
            X_te[f"{col}_TE_auto"] = enc_auto_te[:, i].astype('float32')

            X_tr[f"{col}_TE_10"] = enc_10_tr[:, i].astype('float32')
            X_va[f"{col}_TE_10"] = enc_10_va[:, i].astype('float32')
            X_te[f"{col}_TE_10"] = enc_10_te[:, i].astype('float32')

            X_tr[f"{col}_TE_100"] = enc_100_tr[:, i].astype('float32')
            X_va[f"{col}_TE_100"] = enc_100_va[:, i].astype('float32')
            X_te[f"{col}_TE_100"] = enc_100_te[:, i].astype('float32')

        # 1. High-Resolution LightGBM
        print("  [1/2] Entrenando High-Resolution LightGBM (max_bin=1024)...")
        model_lgb = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=32,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.3,
            reg_alpha=0.071,
            reg_lambda=2.0,
            max_bin=1024,
            random_state=42 + fold,
            feature_pre_filter=False,
            metric='auc',
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[early_stopping(stopping_rounds=120, verbose=False)]
        )
        p_val_lgb = model_lgb.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = p_val_lgb
        oof_lgb_rank[val_idx] = rankdata(p_val_lgb) / len(p_val_lgb)

        p_test_lgb = model_lgb.predict_proba(X_te)[:, 1]
        test_preds_lgb_rank += (rankdata(p_test_lgb) / len(p_test_lgb)) / n_splits

        auc_lgb = roc_auc_score(y_val, p_val_lgb)
        print(f"        Fold {fold} LightGBM AUC: {auc_lgb:.6f} (árboles: {model_lgb.best_iteration_})")

        # 2. High-Resolution XGBoost
        print("  [2/2] Entrenando High-Resolution XGBoost (max_bin=1024)...")
        model_xgb = XGBClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.3,
            reg_alpha=0.071,
            reg_lambda=2.0,
            max_bin=1024,
            tree_method='hist',
            eval_metric='auc',
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=120
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            verbose=False
        )
        p_val_xgb = model_xgb.predict_proba(X_va)[:, 1]
        oof_xgb[val_idx] = p_val_xgb
        oof_xgb_rank[val_idx] = rankdata(p_val_xgb) / len(p_val_xgb)

        p_test_xgb = model_xgb.predict_proba(X_te)[:, 1]
        test_preds_xgb_rank += (rankdata(p_test_xgb) / len(p_test_xgb)) / n_splits

        auc_xgb = roc_auc_score(y_val, p_val_xgb)
        print(f"        Fold {fold} XGBoost  AUC: {auc_xgb:.6f} (árboles: {model_xgb.best_iteration})")

        # Ensamble por rango del Fold
        fold_blend = 0.5 * oof_lgb_rank[val_idx] + 0.5 * oof_xgb_rank[val_idx]
        print(f"      --> Fold {fold} Rank Blend AUC: {roc_auc_score(y_val, fold_blend):.6f} (Tiempo fold: {time.time() - fold_start:.1f}s)")

        del X_tr, X_va, X_te
        gc.collect()

    # Métricas Globales OOF
    print("\n" + "=" * 80)
    print(" RESULTADOS GLOBALES DE VALIDACIÓN CRUZADA (OOF ROC-AUC)")
    print("=" * 80)
    total_auc_lgb = roc_auc_score(y_train, oof_lgb)
    total_auc_xgb = roc_auc_score(y_train, oof_xgb)
    print(f"  LightGBM OOF ROC-AUC : {total_auc_lgb:.6f}")
    print(f"  XGBoost  OOF ROC-AUC : {total_auc_xgb:.6f}")

    # Optimización de pesos de ensamble
    best_rank_auc = 0
    best_w = 0.5
    for w in np.linspace(0.1, 0.9, 17):
        curr_oof = w * oof_lgb_rank + (1.0 - w) * oof_xgb_rank
        curr_auc = roc_auc_score(y_train, curr_oof)
        if curr_auc > best_rank_auc:
            best_rank_auc = curr_auc
            best_w = w

    print(f"  Peso Óptimo LightGBM : {best_w:.2f} | Peso Óptimo XGBoost: {1.0 - best_w:.2f}")
    print(f"  [*] SUPER-ENSEMBLE V3 OOF ROC-AUC: {best_rank_auc:.6f}")

    # Generar predicción final para el conjunto de test
    final_test_preds = best_w * test_preds_lgb_rank + (1.0 - best_w) * test_preds_xgb_rank

    sub_df = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': np.clip(final_test_preds, 0.0, 1.0)
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    versioned_sub_path = os.path.join(SUB_DIR, f"submission_v3_top30_{timestamp}_auc_{best_rank_auc:.5f}.csv")
    final_sub_path = os.path.join(os.path.dirname(__file__), "..", "submission.csv")

    sub_df.to_csv(versioned_sub_path, index=False)
    sub_df.to_csv(final_sub_path, index=False)

    print(f"\n[+] Archivos de sumisión guardados exitosamente:")
    print(f"    - Historial : {versioned_sub_path}")
    print(f"    - Principal : {final_sub_path}")
    print(f"    - Total filas: {len(sub_df):,}")
    print(f"    - Rango probabilidades: [{sub_df['Will_Buy_EV'].min():.4f}, {sub_df['Will_Buy_EV'].max():.4f}]")
    print(f"\nTiempo total de ejecución: {(time.time() - start_time) / 60:.2f} minutos.")

    return best_rank_auc

if __name__ == "__main__":
    train_grandmaster_ensemble()
