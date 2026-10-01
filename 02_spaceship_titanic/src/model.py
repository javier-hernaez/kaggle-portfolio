"""
Spaceship Titanic - SOTA Tri-Engine Ensemble (CatBoost + LightGBM + XGBoost)
Features:
- Advanced domain imputation (Group & Family-level HomePlanet, Cabin, CryoSleep, VIP)
- Cabin position normalization (CabinPosInDeck, CabinRegion, DeckSide)
- Granular expense breakdowns (Luxury vs Essential, Amenity percentages, Zero-spending flags)
- Group & family aggregations (Group Cryo rate, Group spending ratios)
- 10-Fold Stratified Cross-Validation with Out-of-Fold (OOF) Ensembling and Test Bagging
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")
ROOT_SUB = os.path.join(os.path.dirname(__file__), "..", "submission.csv")


def preprocess_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    n_train = len(train)
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # 1. Parse PassengerId
    df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['GroupNum'] = df['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # 2. Parse Name -> LastName
    df['LastName'] = df['Name'].str.split().str[-1]
    df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count').fillna(1)
    
    # 3. Parse Cabin
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    df['Side'] = cabin_split[2]
    
    # --- DOMAIN HEURISTICS & IMPUTATION ---
    expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Children < 13 cannot spend
    for col in expenses:
        df.loc[df['Age'] < 13, col] = 0
        df.loc[df['CryoSleep'] == True, col] = 0
        df[col] = df[col].fillna(0)
        
    df['TotalExpenses'] = df[expenses].sum(axis=1)
    df['NoExpenses'] = (df['TotalExpenses'] == 0).astype(int)
    df.loc[df['TotalExpenses'] > 0, 'CryoSleep'] = df.loc[df['TotalExpenses'] > 0, 'CryoSleep'].fillna(False)
    
    # Impute HomePlanet from GroupId and LastName
    group_hp = df.groupby('GroupId')['HomePlanet'].transform(lambda s: s.ffill().bfill())
    df['HomePlanet'] = df['HomePlanet'].fillna(group_hp)
    last_hp = df.groupby('LastName')['HomePlanet'].transform(lambda s: s.ffill().bfill())
    df['HomePlanet'] = df['HomePlanet'].fillna(last_hp)
    df.loc[(df['HomePlanet'].isna()) & (df['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
    df.loc[(df['HomePlanet'].isna()) & (df['Deck'] == 'G'), 'HomePlanet'] = 'Earth'
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    
    # VIP imputation: Earth passengers are never VIP
    df.loc[df['HomePlanet'] == 'Earth', 'VIP'] = df.loc[df['HomePlanet'] == 'Earth', 'VIP'].fillna(False)
    df.loc[df['CryoSleep'] == True, 'VIP'] = df.loc[df['CryoSleep'] == True, 'VIP'].fillna(False)
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    
    # CryoSleep group imputation
    group_cryo = df.groupby('GroupId')['CryoSleep'].transform(lambda s: s.ffill().bfill())
    df['CryoSleep'] = df['CryoSleep'].fillna(group_cryo).fillna(False).astype(int)
    
    # Cabin group imputation
    group_deck = df.groupby('GroupId')['Deck'].transform(lambda s: s.ffill().bfill())
    df['Deck'] = df['Deck'].fillna(group_deck).fillna('Missing')
    group_side = df.groupby('GroupId')['Side'].transform(lambda s: s.ffill().bfill())
    df['Side'] = df['Side'].fillna(group_side).fillna('Missing')
    group_cabnum = df.groupby('GroupId')['CabinNum'].transform(lambda s: s.ffill().bfill())
    df['CabinNum'] = df['CabinNum'].fillna(group_cabnum).fillna(-1)
    
    # Destination imputation
    group_dest = df.groupby('GroupId')['Destination'].transform(lambda s: s.ffill().bfill())
    df['Destination'] = df['Destination'].fillna(group_dest).fillna('TRAPPIST-1e')
    
    # Age imputation
    df['Age'] = df['Age'].fillna(df.groupby(['HomePlanet', 'Deck'])['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # --- ADVANCED FEATURE ENGINEERING ---
    df['IsChild'] = (df['Age'] < 13).astype(int)
    df['IsSenior'] = (df['Age'] > 55).astype(int)
    df['CabinRegion'] = (df['CabinNum'] // 300).astype(int)
    df['DeckSide'] = df['Deck'] + '_' + df['Side']
    
    # Cabin relative position inside its deck
    max_deck_cabin = df.groupby('Deck')['CabinNum'].transform('max')
    df['CabinPosInDeck'] = (df['CabinNum'] / (max_deck_cabin + 1.0)).fillna(0.0)
    
    # Spending aggregates
    df['LuxurySpend'] = df['Spa'] + df['VRDeck'] + df['RoomService']
    df['EssentialSpend'] = df['FoodCourt'] + df['ShoppingMall']
    df['AmenitiesCount'] = (df[expenses] > 0).sum(axis=1)
    
    for col in expenses + ['TotalExpenses', 'LuxurySpend', 'EssentialSpend']:
        df[f'{col}_log'] = np.log1p(df[col])
    for col in expenses:
        df[f'{col}_pct'] = df[col] / (df['TotalExpenses'] + 1.0)
        df[f'Has_{col}'] = (df[col] > 0).astype(int)
        
    df['SpendPerAge'] = df['TotalExpenses'] / (df['Age'] + 1.0)
    df['GroupTotalExpense'] = df.groupby('GroupId')['TotalExpenses'].transform('sum')
    df['GroupMeanExpense'] = df.groupby('GroupId')['TotalExpenses'].transform('mean')
    df['GroupExpenseRatio'] = df['TotalExpenses'] / (df['GroupTotalExpense'] + 1.0)
    df['GroupCryoCount'] = df.groupby('GroupId')['CryoSleep'].transform('sum')
    df['GroupCryoRate'] = df.groupby('GroupId')['CryoSleep'].transform('mean')
    
    # Encode categorical features
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckSide']
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
        
    features = [
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'IsChild', 'IsSenior',
        'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log',
        'TotalExpenses_log', 'LuxurySpend_log', 'EssentialSpend_log', 'NoExpenses',
        'AmenitiesCount', 'SpendPerAge',
        'Has_RoomService', 'Has_FoodCourt', 'Has_ShoppingMall', 'Has_Spa', 'Has_VRDeck',
        'RoomService_pct', 'FoodCourt_pct', 'ShoppingMall_pct', 'Spa_pct', 'VRDeck_pct',
        'GroupSize', 'IsAlone', 'FamilySize',
        'GroupTotalExpense', 'GroupMeanExpense', 'GroupExpenseRatio',
        'GroupCryoCount', 'GroupCryoRate',
        'Deck', 'CabinNum', 'Side', 'CabinRegion', 'DeckSide', 'CabinPosInDeck'
    ]
    
    X_train = df.iloc[:n_train][features].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_train = df.iloc[:n_train]['Transported'].astype(int).copy()
    X_test = df.iloc[n_train:][features].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_ids = df.iloc[n_train:]['PassengerId'].values
    
    return X_train, y_train, X_test, test_ids, features


def main():
    print("=" * 70)
    print(" SPACESHIP TITANIC: 10-FOLD SOTA TRI-ENGINE ENSEMBLE PIPELINE")
    print("=" * 70)
    
    X_train, y_train, X_test, test_ids, features = preprocess_data()
    n_train = len(X_train)
    n_test = len(X_test)
    n_splits = 10
    print(f"\n[+] Total muestras train: {n_train} | test: {n_test}")
    print(f"[+] Variables de ingenieria ({len(features)}): {features}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_cb = np.zeros(n_train)
    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)
    
    test_preds_cb = np.zeros(n_test)
    test_preds_lgb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    
    print("\n--- Entrenando 10 Folds x 3 Motores (CatBoost, LightGBM, XGBoost) ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # 1. CatBoost
        cb = CatBoostClassifier(
            iterations=700,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            verbose=0
        )
        cb.fit(X_tr, y_tr)
        val_cb = cb.predict_proba(X_va)[:, 1]
        oof_cb[val_idx] = val_cb
        test_preds_cb += cb.predict_proba(X_test)[:, 1] / n_splits
        
        # 2. LightGBM
        lgb = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        val_lgb = lgb.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = val_lgb
        test_preds_lgb += lgb.predict_proba(X_test)[:, 1] / n_splits
        
        # 3. XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42 + fold
        )
        xgb.fit(X_tr, y_tr)
        val_xgb = xgb.predict_proba(X_va)[:, 1]
        oof_xgb[val_idx] = val_xgb
        test_preds_xgb += xgb.predict_proba(X_test)[:, 1] / n_splits
        
        fold_blend = 0.50 * val_cb + 0.30 * val_xgb + 0.20 * val_lgb
        print(f"  Fold {fold+1:02d}/{n_splits:02d} -> CB: {accuracy_score(y_va, val_cb > 0.5):.4f} | LGB: {accuracy_score(y_va, val_lgb > 0.5):.4f} | XGB: {accuracy_score(y_va, val_xgb > 0.5):.4f} | Blend: {accuracy_score(y_va, fold_blend > 0.485):.4f}")
        
    print("\n" + "=" * 70)
    print(" METRICAS GLOBALES OUT-OF-FOLD (OOF)")
    print("=" * 70)
    print(f"  CatBoost OOF Accuracy : {accuracy_score(y_train, oof_cb > 0.5):.5f} | AUC: {roc_auc_score(y_train, oof_cb):.5f}")
    print(f"  LightGBM OOF Accuracy : {accuracy_score(y_train, oof_lgb > 0.5):.5f} | AUC: {roc_auc_score(y_train, oof_lgb):.5f}")
    print(f"  XGBoost  OOF Accuracy : {accuracy_score(y_train, oof_xgb > 0.5):.5f} | AUC: {roc_auc_score(y_train, oof_xgb):.5f}")
    
    # Optimal Ensemble Blend
    oof_blend = 0.50 * oof_cb + 0.30 * oof_xgb + 0.20 * oof_lgb
    test_blend = 0.50 * test_preds_cb + 0.30 * test_preds_xgb + 0.20 * test_preds_lgb
    
    # Threshold fine-tuning
    best_th = 0.5
    best_acc = 0.0
    for th in np.arange(0.46, 0.54, 0.005):
        acc = accuracy_score(y_train, oof_blend > th)
        if acc > best_acc:
            best_acc = acc
            best_th = round(th, 3)
            
    print(f"\n[*] ENSEMBLE GLOBAL BLEND OOF ACCURACY: {best_acc:.5f} (Umbral optimo: {best_th})")
    print(f"[*] Mejora respecto a baseline inicial (0.8110): +{(best_acc - 0.8110):.5f} (+{(best_acc - 0.8110)*100:.2f}%)")
    
    # Generate test submission with 10-Fold Bagging
    test_preds_bool = (test_blend > best_th).astype(bool)
    
    os.makedirs(SUB_DIR, exist_ok=True)
    sub_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': test_preds_bool
    })
    
    versioned_sub = os.path.join(SUB_DIR, "submission_tri_engine_10f.csv")
    sub_df.to_csv(versioned_sub, index=False)
    sub_df.to_csv(ROOT_SUB, index=False)
    
    print(f"\n[+] Archivos de prediccion guardados exitosamente:")
    print(f"    - {versioned_sub}")
    print(f"    - {ROOT_SUB}")
    print(f"[+] Total filas: {len(sub_df)}")
    print(f"[+] Distribucion de prediccion: {sub_df['Transported'].value_counts().to_dict()} (Ratio: {sub_df['Transported'].mean():.1%})")


if __name__ == "__main__":
    main()
