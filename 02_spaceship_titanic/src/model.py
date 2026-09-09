"""
Spaceship Titanic - Advanced Ensemble Model (LightGBM + CatBoost)
Features:
- Domain-specific logic (CryoSleep vs Amenities Spending)
- Deck, CabinNum, Side parsing with Deck-HomePlanet correlation
- Group sizing from PassengerId
- Log transformations on heavy spenders
- 5-Fold Stratified Cross-Validation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")

def preprocess_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    n_train = len(train)
    data = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # 1. Información de grupo a partir de PassengerId (gggg_pp)
    data['GroupId'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
    data['GroupNum'] = data['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    data['GroupSize'] = data.groupby('GroupId')['GroupId'].transform('count')
    data['IsAlone'] = (data['GroupSize'] == 1).astype(int)
    
    # 2. Desglose de Cabina (Deck / Num / Side)
    cabin_split = data['Cabin'].str.split('/', expand=True)
    data['Deck'] = cabin_split[0].fillna('Missing')
    data['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce').fillna(-1)
    data['Side'] = cabin_split[2].fillna('Missing')
    
    # 3. Lógica de Gastos vs Criocongelación (CryoSleep)
    expenses = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Si estaba en CryoSleep, no pudo gastar nada en servicios
    for col in expenses:
        data.loc[data['CryoSleep'] == True, col] = 0
        data[col] = data[col].fillna(0)
        
    data['TotalExpenses'] = data[expenses].sum(axis=1)
    data['NoExpenses'] = (data['TotalExpenses'] == 0).astype(int)
    
    # Si gastó dinero, con certeza NO estaba en CryoSleep
    data.loc[data['TotalExpenses'] > 0, 'CryoSleep'] = data.loc[data['TotalExpenses'] > 0, 'CryoSleep'].fillna(False)
    data['CryoSleep'] = data['CryoSleep'].fillna(False).astype(int)
    data['VIP'] = data['VIP'].fillna(False).astype(int)
    
    # Log-transformación de gastos para suavizar asimetría
    for col in expenses + ['TotalExpenses']:
        data[f'{col}_log'] = np.log1p(data[col])
        
    # 4. Imputación inteligente de Planeta de Origen según la Cubierta
    # Cubiertas A, B, C, T corresponden a pasajeros de Europa
    # Cubierta G corresponde a pasajeros de la Tierra
    data.loc[(data['HomePlanet'].isna()) & (data['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
    data.loc[(data['HomePlanet'].isna()) & (data['Deck'] == 'G'), 'HomePlanet'] = 'Earth'
    data['HomePlanet'] = data['HomePlanet'].fillna('Earth')
    data['Destination'] = data['Destination'].fillna('TRAPPIST-1e')
    
    # 5. Edad
    data['Age'] = data['Age'].fillna(data.groupby('HomePlanet')['Age'].transform('median'))
    data['IsChild'] = (data['Age'] < 13).astype(int)
    
    # 6. Codificación de variables categóricas
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    for c in cat_cols:
        data[c] = data[c].astype('category').cat.codes
        
    features = [
        'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'IsChild',
        'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log',
        'TotalExpenses_log', 'NoExpenses', 'GroupSize', 'IsAlone',
        'Deck', 'CabinNum', 'Side'
    ]
    
    X_train = data.iloc[:n_train][features]
    y_train = data.iloc[:n_train]['Transported'].astype(int)
    X_test = data.iloc[n_train:][features]
    test_ids = data.iloc[n_train:]['PassengerId']
    
    return X_train, y_train, X_test, test_ids, features

def main():
    print("=" * 65)
    print(" MODELO AVANZADO SPACESHIP TITANIC: LIGHTGBM + CATBOOST")
    print("=" * 65)
    
    X_train, y_train, X_test, test_ids, features = preprocess_data()
    print(f"\n[+] Variables seleccionadas ({len(features)}): {features}")
    
    # Modelos base con hiperparámetros afinados
    lgb = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    cb = CatBoostClassifier(
        iterations=350,
        depth=5,
        learning_rate=0.03,
        random_seed=42,
        verbose=0
    )
    
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb), ('cb', cb)],
        voting='soft',
        weights=[1, 1]
    )
    
    # Validación Cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n--- Rendimiento en Validación Cruzada (5 Folds) ---")
    for name, model in [('LightGBM', lgb), ('CatBoost', cb), ('Ensemble (LGB+CB)', ensemble)]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"  {name:20s}: Exactitud Media = {scores.mean():.4f} (+/- {scores.std():.4f}) | Folds: {np.round(scores, 3)}")
        
    print("\n[+] Entrenando el ensemble final en todos los datos de entrenamiento...")
    ensemble.fit(X_train, y_train)
    
    # Predicciones para test
    preds = ensemble.predict(X_test)
    preds_bool = preds.astype(bool)
    
    sub_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': preds_bool
    })
    
    out_file = os.path.join(SUB_DIR, "submission_ensemble.csv")
    root_file = os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    
    sub_df.to_csv(out_file, index=False)
    sub_df.to_csv(root_file, index=False)
    
    print(f"\n[+] Predicciones guardadas con éxito en:")
    print(f"    - {out_file}")
    print(f"    - {root_file}")
    print(f"\nDistribución predicha:\n{sub_df['Transported'].value_counts().to_dict()} (Transportados: {sub_df['Transported'].mean():.1%})")

if __name__ == "__main__":
    main()
