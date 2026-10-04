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
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize

DATA_DIR = r"c:\Users\Javier\Desktop\kaggle\03_house_prices\data"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Ames outlier removal
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
y_train = np.log1p(train['SalePrice']).values
n_train = len(train)
n_test = len(test)

data = pd.concat([train.drop(columns=['SalePrice']), test], sort=False).reset_index(drop=True)

# 1. Quality Ordinal Mappings
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

# 2. Rich Domain Interactions
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

data['OverallGrade'] = data['OverallQual'] * data['OverallCond']
data['QualSF'] = data['OverallQual'] * data['TotalSF']
data['ExterGrade'] = data['ExterQual_ord'] * data['ExterCond_ord']
data['KitchenGrade'] = data['KitchenQual_ord'] * data['KitchenAbvGr'].fillna(1)
data['GarageGrade'] = data['GarageQual_ord'] * data['GarageCars'].fillna(0)

# Amenities binary flags
data['HasPool'] = (data['PoolArea'].fillna(0) > 0).astype(int)
data['HasGarage'] = (data['GarageArea'].fillna(0) > 0).astype(int)
data['HasBsmt'] = (data['TotalBsmtSF'].fillna(0) > 0).astype(int)
data['HasFireplace'] = (data['Fireplaces'].fillna(0) > 0).astype(int)
data['Has2ndFloor'] = (data['2ndFlrSF'].fillna(0) > 0).astype(int)

# Log transform heavily skewed numeric features
skewed_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 
               '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'TotalSF', 'TotalPorch']
for c in skewed_cols:
    data[f'{c}_log'] = np.log1p(data[c].fillna(0).clip(lower=0))

# Imputation
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

print(f"Features created: {X_train.shape[1]}")

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

models = {
    'Ridge': make_pipeline(RobustScaler(), Ridge(alpha=15.0)),
    'Lasso': make_pipeline(RobustScaler(), Lasso(alpha=0.00045, max_iter=15000, random_state=42)),
    'ElasticNet': make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=15000, random_state=42)),
    'LightGBM': LGBMRegressor(n_estimators=700, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=900, depth=4, learning_rate=0.02, random_seed=42, verbose=0),
    'XGBoost': XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, random_state=42),
    'GBR': GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.02, subsample=0.8, random_state=42)
}

oof_preds = {name: np.zeros(n_train) for name in models}
test_preds = {name: np.zeros(n_test) for name in models}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = model.predict(X_va)
        test_preds[name] += model.predict(X_test) / n_splits

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\n--- OOF RMSLE Individual Models ---")
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

print("\n--- Optimal Ensembling Weights ---")
for name, w in zip(models.keys(), opt_weights):
    print(f"  {name:12s}: {w:.4f}")

opt_oof_rmse = rmse(y_train, OOF_matrix @ opt_weights)
print(f"\n[*] OPTIMAL STACKED OOF RMSLE: {opt_oof_rmse:.5f} (Mejora: -{(0.12589 - opt_oof_rmse):.5f})")

final_log_preds = TEST_matrix @ opt_weights
final_sale_prices = np.expm1(final_log_preds)

sub_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_sale_prices})
sub_file = r"c:\Users\Javier\Desktop\kaggle\03_house_prices\submission.csv"
ver_file = r"c:\Users\Javier\Desktop\kaggle\03_house_prices\submissions\submission_sota_7engine_10f.csv"
os.makedirs(os.path.dirname(ver_file), exist_ok=True)
sub_df.to_csv(sub_file, index=False)
sub_df.to_csv(ver_file, index=False)

print(f"\nPredicciones guardadas exitosamente en:\n  - {sub_file}\n  - {ver_file}")
