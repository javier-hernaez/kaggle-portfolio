"""
Baseline Model for Titanic: Machine Learning from Disaster
Uses RandomForestClassifier with cross-validation and feature engineering.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, test_df

def preprocess_features(df, is_train=True):
    # Create a copy
    data = df.copy()
    
    # 1. Feature Engineering: Title from Name
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')
    data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    data['Title'] = data['Title'].map(title_mapping).fillna(0).astype(int)
    
    # 2. Gender mapping
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
    # 3. Family Size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # 4. Impute missing Age with median
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    # 5. Impute missing Fare with median
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # 6. Impute missing Embarked with mode & map
    mode_embarked = data['Embarked'].mode()[0] if not data['Embarked'].mode().empty else 'S'
    data['Embarked'] = data['Embarked'].fillna(mode_embarked)
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    data['Embarked'] = data['Embarked'].map(embarked_mapping).fillna(0).astype(int)
    
    # Select feature columns
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    
    X = data[features]
    y = data['Survived'] if is_train else None
    passenger_ids = data['PassengerId']
    
    return X, y, passenger_ids

def main():
    print("--- 1. Cargando datos ---")
    train_df, test_df = load_data()
    print(f"Entrenamiento: {train_df.shape[0]} muestras")
    print(f"Test: {test_df.shape[0]} muestras")
    
    print("\n--- 2. Preprocesamiento e Ingeniería de Características ---")
    X_train, y_train, _ = preprocess_features(train_df, is_train=True)
    X_test, _, test_ids = preprocess_features(test_df, is_train=False)
    
    print(f"Variables utilizadas ({len(X_train.columns)}): {list(X_train.columns)}")
    
    print("\n--- 3. Entrenamiento y Validación Cruzada (Random Forest) ---")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"Puntuaciones de CV (5 folds): {np.round(scores, 4)}")
    print(f"Exactitud media (Mean Accuracy): {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Entrenar con todo el conjunto de entrenamiento
    model.fit(X_train, y_train)
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nImportancia de las variables:")
    for _, row in feature_importance.iterrows():
        print(f"  - {row['Feature']:12s}: {row['Importance']:.4f}")
        
    print("\n--- 4. Generando Predicciones para Test ---")
    predictions = model.predict(X_test)
    
    submission_path = os.path.join(SUB_DIR, "submission_baseline.csv")
    submission_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    submission_df.to_csv(submission_path, index=False)
    
    print(f"¡Archivo de predicción generado con éxito en:\n{submission_path}")
    print("\nVista previa de la sumisión:")
    print(submission_df.head(10))
    print(f"\nDistribución de supervivientes predichos: {submission_df['Survived'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
