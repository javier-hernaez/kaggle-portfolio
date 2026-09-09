"""
Análisis Exploratorio de Datos (EDA) para el Titanic
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def run_eda():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print("=" * 60)
    print("1. RESUMEN GENERAL")
    print("=" * 60)
    print(f"Filas de entrenamiento : {train.shape[0]}, Columnas: {train.shape[1]}")
    print(f"Filas de test          : {test.shape[0]}, Columnas: {test.shape[1]}")
    
    print("\n" + "=" * 60)
    print("2. VALORES NULOS / FALTANTES")
    print("=" * 60)
    print("En Train:")
    print(train.isnull().sum()[train.isnull().sum() > 0])
    print("\nEn Test:")
    print(test.isnull().sum()[test.isnull().sum() > 0])
    
    print("\n" + "=" * 60)
    print("3. TASA DE SUPERVIVENCIA POR CARACTERÍSTICA")
    print("=" * 60)
    
    print("\nPor Sexo:")
    print(train.groupby('Sex')['Survived'].agg(['count', 'mean']))
    
    print("\nPor Clase (Pclass):")
    print(train.groupby('Pclass')['Survived'].agg(['count', 'mean']))
    
    print("\nPor Sexo y Clase:")
    print(train.pivot_table('Survived', index='Sex', columns='Pclass', aggfunc=['count', 'mean']))
    
    print("\nPor Puerto de Embarque (Embarked):")
    print(train.groupby('Embarked')['Survived'].agg(['count', 'mean']))

if __name__ == "__main__":
    run_eda()
