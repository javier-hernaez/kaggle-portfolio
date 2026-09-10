"""
Exploratory Data Analysis for Playground Series s6e9:
Predicting Electric Vehicle Purchases
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def run_eda():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)
    
    print("=" * 70)
    print("RESUMEN GENERAL DEL DATASET")
    print("=" * 70)
    print(f"Dimensiones Train : {train.shape}")
    print(f"Dimensiones Test  : {test.shape}")
    print(f"Dimensiones Sub   : {sub.shape}")
    
    print("\nColumnas en Train:")
    for col in train.columns:
        n_unique = train[col].nunique()
        n_null = train[col].isnull().sum()
        dtype = train[col].dtype
        print(f"  - {col:30s} | Tipo: {str(dtype):10s} | Únicos: {n_unique:8d} | Nulos: {n_null}")

    print("\nDistribución del Target ('Will_Buy_EV'):")
    target_counts = train['Will_Buy_EV'].value_counts()
    target_prop = train['Will_Buy_EV'].value_counts(normalize=True)
    for val, count in target_counts.items():
        print(f"  Clase {val}: {count:,} ({target_prop[val]:.2%})")

    print("\nPrimeras 5 filas del Train:")
    print(train.head(5))

    print("\nEstadísticas descriptivas de variables numéricas:")
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    print(train[num_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nVariables categóricas ({len(cat_cols)}):")
    for col in cat_cols:
        print(f"  {col}: {train[col].value_counts().to_dict()}")

    # Correlaciones con Will_Buy_EV
    train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
    analysis_cols = [c for c in num_cols if c != 'id'] + ['target']
    corr = train[analysis_cols].corr()['target'].sort_values(ascending=False)
    print("\nCorrelación de Pearson con target (Will_Buy_EV):")
    print(corr)

    print("\nTasa de compra por variable categórica:")
    for col in [c for c in cat_cols if c != 'Will_Buy_EV']:
        rates = train.groupby(col)['target'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        rates['mean'] = rates['mean'].map(lambda x: f"{x:.2%}")
        print(f"\n-- {col} --")
        print(rates)

if __name__ == "__main__":
    run_eda()
