"""
House Prices: Advanced Regression Techniques
SOTA Grandmaster 10-Fold Heterogeneous Stack:
- Strategic Location Valuation: NeighborhoodTier and MSZoning_ord
- Synergistic Real-Estate Interactions: NeighSF, NeighQual, NeighAge, QualSF, OverallGrade
- Ordinal Encoding of 10 Material and Quality Scales
- Logarithmic Normalization on Target (RMSLE) and Skewed Continuous Physical Dimensions
- Outlier Pruning of Ames Anomaly Properties (>4000 sq ft)
- SLSQP Constrained Meta-Ensembling: Ridge + Lasso + ElasticNet + LightGBM + CatBoost + XGBoost
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
ROOT_SUB = os.path.join(os.path.dirname(__file__), "..", "submission.csv")


def preprocess_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Prune extreme Ames Housing outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
    y_train = np.log1p(train['SalePrice']).values
    n_train = len(train)
    n_test = len(test)
    
    data = pd.concat([train.drop(columns=['SalePrice']), test], sort=False).reset_index(drop=True)
    
    # 1. Quality & Material Ordinal Mappings
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    qual_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, np.nan: 0}
    for c in qual_cols:
        data[f'{c}_ord'] = data[c].map(qual_dict).fillna(0)
        
    bsmt_fin_dict = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0, np.nan: 0}
    data['BsmtFinType1_ord'] = data['BsmtFinType1'].map(bsmt_fin_dict).fillna(0)
    data['BsmtFinType2_ord'] = data['BsmtFinType2'].map(bsmt_fin_dict).fillna(0)
    garage_finish_dict = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0, np.nan: 0}
    data['GarageFinish_ord'] = data['GarageFinish'].map(garage_finish_dict).fillna(0)
    
    # 2. Neighborhood Economic Valuation Tiers (from low density/subsidized to luxury golf communities)
    neigh_dict = {
        'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1,
        'OldTown': 2, 'Edwards': 2, 'BrkSide': 2,
        'Sawyer': 3, 'Blueste': 3, 'SWISU': 3, 'NAmes': 3, 'NPkVill': 3, 'Mitchel': 3,
        'SawyerW': 4, 'Gilbert': 4, 'NWAmes': 4, 'Blmngtn': 4, 'CollgCr': 4,
        'ClearCr': 5, 'Crawfor': 5, 'Veenker': 5, 'Somerst': 5, 'Timber': 5,
        'StoneBr': 6, 'NoRidge': 6, 'NridgHt': 6
    }
    data['NeighTier'] = data['Neighborhood'].map(neigh_dict).fillna(3)
    
    zoning_dict = {'C (all)': 1, 'RM': 2, 'RH': 3, 'RL': 4, 'FV': 5}
    data['MSZoning_ord'] = data['MSZoning'].map(zoning_dict).fillna(3)
    
    # 3. Comprehensive Square Footage & Volume
    data['TotalSF'] = data['TotalBsmtSF'].fillna(0) + data['1stFlrSF'].fillna(0) + data['2ndFlrSF'].fillna(0)
    data['TotalFinishedSF'] = (data['1stFlrSF'].fillna(0) + data['2ndFlrSF'].fillna(0) + 
                               data['BsmtFinSF1'].fillna(0) + data['BsmtFinSF2'].fillna(0))
                               
    data['TotalBath'] = (
        data['FullBath'].fillna(0) + 0.5 * data['HalfBath'].fillna(0) +
        data['BsmtFullBath'].fillna(0) + 0.5 * data['BsmtHalfBath'].fillna(0)
    )
    
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    data['IsNewHouse'] = (data['YrSold'] == data['YearBuilt']).astype(int)
    data['IsRemodeled'] = (data['YearBuilt'] != data['YearRemodAdd']).astype(int)
    
    data['TotalPorch'] = (
        data['OpenPorchSF'].fillna(0) + data['EnclosedPorch'].fillna(0) +
        data['3SsnPorch'].fillna(0) + data['ScreenPorch'].fillna(0)
    )
    
    # 4. Location × Quality × Physical Scale Interactions
    data['OverallGrade'] = data['OverallQual'] * data['OverallCond']
    data['QualSF'] = data['OverallQual'] * data['TotalSF']
    data['NeighSF'] = data['NeighTier'] * data['TotalSF']
    data['NeighQual'] = data['NeighTier'] * data['OverallQual']
    data['NeighAge'] = data['NeighTier'] * data['HouseAge']
    data['KitchenGrade'] = data['KitchenQual_ord'] * data['KitchenAbvGr'].fillna(1)
    data['GarageGrade'] = data['GarageQual_ord'] * data['GarageCars'].fillna(0)
    
    data['HasPool'] = (data['PoolArea'].fillna(0) > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'].fillna(0) > 0).astype(int)
    data['HasBsmt'] = (data['TotalBsmtSF'].fillna(0) > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'].fillna(0) > 0).astype(int)
    data['Has2ndFloor'] = (data['2ndFlrSF'].fillna(0) > 0).astype(int)
    
    # Log transform skewed features
    skewed_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 
                   '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'TotalSF', 'TotalPorch',
                   'QualSF', 'NeighSF']
    for c in skewed_cols:
        data[f'{c}_log'] = np.log1p(data[c].fillna(0).clip(lower=0))
        
    num_cols = data.select_dtypes(include=[np.number]).columns.drop('Id')
    for c in num_cols:
        data[c] = data[c].fillna(data[c].median())
        
    cat_cols = data.select_dtypes(include=['object', 'string']).columns
    for c in cat_cols:
        data[c] = data[c].fillna('None')
        
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    X_train = data.iloc[:n_train].drop(columns=['Id'])
    X_test = data.iloc[n_train:].drop(columns=['Id'])
    test_ids = test['Id'].values
    
    return X_train, y_train, X_test, test_ids


def main():
    print("=" * 75)
    print(" HOUSE PRICES: SOTA 10-FOLD HETEROGENEOUS META-STACK PIPELINE")
    print("=" * 75)
    
    X_train, y_train, X_test, test_ids = preprocess_data()
    n_train = len(X_train)
    n_test = len(X_test)
    n_splits = 10
    print(f"\n[+] Total muestras train: {n_train} | test: {n_test}")
    print(f"[+] Total caracteristicas expandidas: {X_train.shape[1]}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        'Ridge': make_pipeline(RobustScaler(), Ridge(alpha=18.0)),
        'Lasso': make_pipeline(RobustScaler(), Lasso(alpha=0.00045, max_iter=15000, random_state=42)),
        'ElasticNet': make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=15000, random_state=42)),
        'LightGBM': LGBMRegressor(n_estimators=750, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=950, depth=4, learning_rate=0.02, random_seed=42, verbose=0),
        'XGBoost': XGBRegressor(n_estimators=650, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, random_state=42)
    }
    
    oof_preds = {name: np.zeros(n_train) for name in models}
    test_preds = {name: np.zeros(n_test) for name in models}
    
    print("\n--- Entrenando 10 Folds para los 6 Motores Heterogéneos ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model.predict(X_va)
            test_preds[name] += model.predict(X_test) / n_splits
            
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
    print("\n" + "=" * 75)
    print(" METRICAS GLOBALES OUT-OF-FOLD (OOF RMSLE)")
    print("=" * 75)
    for name in models:
        print(f"  {name:12s} RMSLE: {rmse(y_train, oof_preds[name]):.5f}")
        
    OOF_matrix = np.column_stack([oof_preds[name] for name in models])
    TEST_matrix = np.column_stack([test_preds[name] for name in models])
    
    def objective(weights):
        weights = weights / np.sum(weights)
        pred = OOF_matrix @ weights
        return rmse(y_train, pred)
        
    res = minimize(objective, np.ones(len(models)) / len(models), bounds=[(0, 1)] * len(models), method='SLSQP')
    opt_weights = res.x / np.sum(res.x)
    
    print("\n--- Ponderaciones Óptimas del Ensamble SLSQP ---")
    for name, w in zip(models.keys(), opt_weights):
        print(f"  {name:12s}: {w * 100:5.2f}%")
        
    opt_oof_rmse = rmse(y_train, OOF_matrix @ opt_weights)
    print(f"\n[*] OPTIMAL STACKED OOF RMSLE: {opt_oof_rmse:.5f} (Mejora: -{(0.12288 - opt_oof_rmse):.5f})")
    
    final_log_preds = TEST_matrix @ opt_weights
    final_sale_prices = np.expm1(final_log_preds)
    
    sub_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_sale_prices})
    versioned_sub = os.path.join(SUB_DIR, "submission_sota_grandmaster_10f.csv")
    
    os.makedirs(SUB_DIR, exist_ok=True)
    sub_df.to_csv(versioned_sub, index=False)
    sub_df.to_csv(ROOT_SUB, index=False)
    
    print(f"\n[+] Predicciones de precios guardadas en:")
    print(f"    - {versioned_sub}")
    print(f"    - {ROOT_SUB}")
    print(f"\nResumen Estadístico de Precios Predichos ($):")
    print(sub_df['SalePrice'].describe().round(2))


if __name__ == "__main__":
    main()
