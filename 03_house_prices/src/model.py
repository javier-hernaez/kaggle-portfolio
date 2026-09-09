"""
House Prices: Advanced Regression Techniques
Stacking / Blending Solution:
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 feature selection)
- CatBoost Regressor
- LightGBM Regressor
Evaluated with Out-Of-Fold (OOF) 5-Fold Cross Validation on RMSLE.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")

def preprocess_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Eliminar valores atípicos extremos documentados en el paper de Ames Housing
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
    
    # Transformación logarítmica de la variable objetivo (métrica oficial RMSLE)
    y_train = np.log1p(train['SalePrice'])
    n_train = len(train)
    
    data = pd.concat([train.drop(columns=['SalePrice']), test], sort=False).reset_index(drop=True)
    
    # 1. Ingeniería de Características
    # Superficie habitable y construida total
    data['TotalSF'] = data['TotalBsmtSF'].fillna(0) + data['1stFlrSF'].fillna(0) + data['2ndFlrSF'].fillna(0)
    
    # Número total de baños (ponderando medios baños)
    data['TotalBath'] = (
        data['FullBath'].fillna(0) + 0.5 * data['HalfBath'].fillna(0) +
        data['BsmtFullBath'].fillna(0) + 0.5 * data['BsmtHalfBath'].fillna(0)
    )
    
    # Antigüedad en el momento de venta
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    
    # Superficie total de porches
    data['TotalPorch'] = (
        data['OpenPorchSF'].fillna(0) + data['EnclosedPorch'].fillna(0) +
        data['3SsnPorch'].fillna(0) + data['ScreenPorch'].fillna(0)
    )
    
    # Indicadores booleanos de comodidades
    data['HasPool'] = (data['PoolArea'].fillna(0) > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'].fillna(0) > 0).astype(int)
    data['HasBsmt'] = (data['TotalBsmtSF'].fillna(0) > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'].fillna(0) > 0).astype(int)
    
    # 2. Imputación de nulos
    num_cols = data.select_dtypes(include=[np.number]).columns.drop('Id')
    cat_cols = data.select_dtypes(include=['object', 'string']).columns
    
    for c in num_cols:
        data[c] = data[c].fillna(data[c].median())
        
    # One-Hot Encoding para categóricas
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    X_train = data.iloc[:n_train].drop(columns=['Id'])
    X_test = data.iloc[n_train:].drop(columns=['Id'])
    test_ids = test['Id']
    
    return X_train, y_train, X_test, test_ids

def main():
    print("=" * 65)
    print(" MODELO AVANZADO HOUSE PRICES: BLEND (RIDGE + LASSO + CB + LGB)")
    print("=" * 65)
    
    X_train, y_train, X_test, test_ids = preprocess_data()
    n_train = len(X_train)
    print(f"\n[+] Muestras de Train: {n_train}, Variables generadas: {X_train.shape[1]}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Definición de modelos
    ridge = make_pipeline(RobustScaler(), Ridge(alpha=15.0))
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=10000, random_state=42))
    lgb = LGBMRegressor(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    cb = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.03, random_seed=42, verbose=0)
    
    oof_ridge = np.zeros(n_train)
    oof_lasso = np.zeros(n_train)
    oof_lgb = np.zeros(n_train)
    oof_cb = np.zeros(n_train)
    
    test_preds_ridge = np.zeros(len(X_test))
    test_preds_lasso = np.zeros(len(X_test))
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cb = np.zeros(len(X_test))
    
    print("\n--- Entrenando y Evaluando con Validación Cruzada (5 Folds) ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        ridge.fit(X_tr, y_tr)
        oof_ridge[val_idx] = ridge.predict(X_va)
        test_preds_ridge += ridge.predict(X_test) / 5
        
        lasso.fit(X_tr, y_tr)
        oof_lasso[val_idx] = lasso.predict(X_va)
        test_preds_lasso += lasso.predict(X_test) / 5
        
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb.predict(X_va)
        test_preds_lgb += lgb.predict(X_test) / 5
        
        cb.fit(X_tr, y_tr)
        oof_cb[val_idx] = cb.predict(X_va)
        test_preds_cb += cb.predict(X_test) / 5
        
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
    print(f"  Ridge RMSLE   : {rmse(y_train, oof_ridge):.4f}")
    print(f"  Lasso RMSLE   : {rmse(y_train, oof_lasso):.4f}")
    print(f"  LightGBM RMSLE: {rmse(y_train, oof_lgb):.4f}")
    print(f"  CatBoost RMSLE: {rmse(y_train, oof_cb):.4f}")
    
    # Ensamble ponderado de predicciones
    oof_blend = 0.35 * oof_ridge + 0.25 * oof_lasso + 0.25 * oof_cb + 0.15 * oof_lgb
    blend_rmsle = rmse(y_train, oof_blend)
    print(f"\n[*] BLEND OOF RMSLE TOTAL: {blend_rmsle:.4f}")
    
    test_blend_log = (
        0.35 * test_preds_ridge + 
        0.25 * test_preds_lasso + 
        0.25 * test_preds_cb + 
        0.15 * test_preds_lgb
    )
    
    # Revertir la transformación logarítmica a dólares reales
    final_sale_prices = np.expm1(test_blend_log)
    
    sub_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': final_sale_prices
    })
    
    out_file = os.path.join(SUB_DIR, "submission_blend.csv")
    root_file = os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    
    sub_df.to_csv(out_file, index=False)
    sub_df.to_csv(root_file, index=False)
    
    print(f"\n[+] Predicciones de precios guardadas en:\n    - {out_file}\n    - {root_file}")
    print("\nResumen de Precios Predichos ($):")
    print(sub_df['SalePrice'].describe().round(2))

if __name__ == "__main__":
    main()
