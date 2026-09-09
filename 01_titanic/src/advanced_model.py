"""
Advanced Titanic ML Model
Implements:
1. Woman-Child-Group & Family/Ticket Survival Analysis (WCG)
2. Advanced Feature Engineering (Title, Deck, Fare/Person, FamilySize, Imputations)
3. Multi-Model Ensemble: CatBoost + LightGBM + Random Forest with Soft Voting
4. Stratified 5-Fold Cross Validation
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")

def build_features():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Combinar para ingeniería coherente de grupos y billetes compartidos
    data = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # 1. Títulos de cortesía limpios
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')
    data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    data['Title_Code'] = data['Title'].map(title_map).fillna(4).astype(int)
    
    # 2. Imputación fina de Edad por (Clase, Sexo, Título)
    data['Age'] = data.groupby(['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    # 3. Billete compartido y Tarifa por persona real
    data['Fare'] = data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'))
    data['Ticket_Count'] = data.groupby('Ticket')['Ticket'].transform('count')
    data['Fare_Per_Person'] = data['Fare'] / data['Ticket_Count']
    
    # 4. Tamaño de familia e indicador de viaje en solitario
    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['Family_Size'] == 1).astype(int)
    
    # 5. Cubierta de camarote (Deck)
    data['Deck'] = data['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'M')
    deck_map = {'M': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 0}
    data['Deck'] = data['Deck'].map(deck_map).fillna(0).astype(int)
    
    # 6. Puerto de Embarque y Género
    data['Embarked'] = data['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data['Sex_Code'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
    # 7. Rangos discretizados (Bins)
    data['Age_Bin'] = pd.qcut(data['Age'], 5, labels=False, duplicates='drop')
    data['Fare_Bin'] = pd.qcut(data['Fare'], 5, labels=False, duplicates='drop')
    
    # 8. Métrica de Supervivencia Familiar / Grupo (Woman-Child Group Feature)
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0].strip())
    data['Family_Survival'] = 0.5  # Neutral por defecto (sin información)
    
    # Buscar supervivencia en familias (mismo Apellido + Tarifa)
    for _, grp_df in data.groupby(['Surname', 'Fare']):
        if len(grp_df) > 1:
            for idx, row in grp_df.iterrows():
                other = grp_df[grp_df['PassengerId'] != row['PassengerId']]['Survived'].dropna()
                if len(other) > 0:
                    if other.max() == 1.0:
                        data.loc[idx, 'Family_Survival'] = 1.0
                    elif other.min() == 0.0:
                        data.loc[idx, 'Family_Survival'] = 0.0
                        
    # Para los que siguen neutros, buscar por billete compartido (amigos, nanas, etc.)
    for _, grp_df in data.groupby('Ticket'):
        if len(grp_df) > 1:
            for idx, row in grp_df.iterrows():
                if data.loc[idx, 'Family_Survival'] == 0.5:
                    other = grp_df[grp_df['PassengerId'] != row['PassengerId']]['Survived'].dropna()
                    if len(other) > 0:
                        if other.max() == 1.0:
                            data.loc[idx, 'Family_Survival'] = 1.0
                        elif other.min() == 0.0:
                            data.loc[idx, 'Family_Survival'] = 0.0

    features = [
        'Pclass', 'Sex_Code', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title_Code',
        'Family_Size', 'IsAlone', 'Fare_Per_Person', 'Deck', 'Age_Bin', 'Fare_Bin', 'Family_Survival'
    ]
    
    X_train = data.iloc[:len(train)][features]
    y_train = data.iloc[:len(train)]['Survived'].astype(int)
    X_test = data.iloc[len(train):][features]
    test_ids = data.iloc[len(train):]['PassengerId'].astype(int)
    
    return X_train, y_train, X_test, test_ids, features

def main():
    print("=" * 65)
    print(" MODELO AVANZADO TITANIC: ENSEMBLE + WOMAN-CHILD-GROUP (WCG)")
    print("=" * 65)
    
    X_train, y_train, X_test, test_ids, features = build_features()
    print(f"\n[+] Características extraídas ({len(features)}): {features}")
    
    # 1. Definición de modelos base
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    lgb = LGBMClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    cb = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.04,
        random_seed=42,
        verbose=0
    )
    
    # Ensemble de votación suave (ponderando probabilidades)
    ensemble = VotingClassifier(
        estimators=[('cb', cb), ('rf', rf), ('lgb', lgb)],
        voting='soft',
        weights=[2, 1, 1]  # Dar algo más de peso a CatBoost por su mejor CV
    )
    
    # 2. Evaluación por Validación Cruzada Estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Rendimiento en Validación Cruzada Estratificada (5 Folds) ---")
    for name, model in [('Random Forest', rf), ('LightGBM', lgb), ('CatBoost', cb), ('Ensemble (WCG)', ensemble)]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"  {name:20s}: Exactitud Media = {scores.mean():.4f} (+/- {scores.std():.4f}) | Folds: {np.round(scores, 3)}")
        
    # 3. Entrenamiento en todo el conjunto de entrenamiento
    print("\n[+] Entrenando el Ensemble definitivo...")
    ensemble.fit(X_train, y_train)
    
    # 4. Generación de predicciones
    test_preds = ensemble.predict(X_test)
    
    sub_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': test_preds
    })
    
    # Guardar en submissions/ y en submission.csv en raíz
    sub_path = os.path.join(SUB_DIR, "submission_advanced_ensemble.csv")
    root_sub_path = os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv(root_sub_path, index=False)
    
    print(f"\n[+] Predicciones guardadas con éxito en:\n    - {sub_path}\n    - {root_sub_path}")
    print(f"\nDistribución de supervivientes predichos:\n{sub_df['Survived'].value_counts().to_dict()} (Tasa: {sub_df['Survived'].mean():.1%})")

if __name__ == "__main__":
    main()
