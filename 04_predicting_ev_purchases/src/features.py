"""
Grandmaster Feature Engineering Pipeline (High-Performance & Memory-Optimized)
Playground Series s6e9: Predicting Electric Vehicle Purchases
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from scipy.special import ndtr

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def build_grandmaster_features():
    print("=" * 75)
    print(" INGENIERÍA DE VARIABLES GRANDMASTER (ALTA VELOCIDAD Y MEMORIA ÓPTIMA)")
    print("=" * 75)
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    orig_path = os.path.join(DATA_DIR, "EV_Adoption_and_Range_Anxiety_Dataset.csv")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)
    test_ids = test['id']
    
    TARGET = 'Will_Buy_EV'
    train[TARGET] = (train[TARGET].astype(str) == 'Yes').astype('int8')
    if TARGET in orig.columns:
        orig[TARGET] = (orig[TARGET].astype(str) == 'Yes').astype('int8')
        
    n_train = len(train)
    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    y_train = train[TARGET]
    del train, test
    gc.collect()
    
    # Diccionario para acumular nuevas variables de forma eficiente
    new_cols = {}
    
    # 1. Variables de dominio previo y Probit DGP
    subsidy_num = (combined['Subsidy_Available'].astype(str) == 'Yes').astype('float32')
    home_charge_num = (combined['Home_Charging_Possible'].astype(str) == 'Yes').astype('float32')
    med_anx = (combined['Range_Anxiety_Level'].astype(str) == 'Medium').astype('float32')
    high_anx = (combined['Range_Anxiety_Level'].astype(str) == 'High').astype('float32')
    income_scaled = (combined['Annual_Income_USD'] / 100000.0).astype('float32')
    
    new_cols['DGP_Probit_Score'] = (
        1.2 * income_scaled
        + 0.6 * combined['Environmental_Concern_Level'].astype('float32')
        + 2.0 * subsidy_num
        - 1.0 * med_anx
        - 3.0 * high_anx
        - 5.5
    )
    new_cols['DGP_Probit_Prob'] = ndtr(new_cols['DGP_Probit_Score']).astype('float32')
    
    # 2. Resolución de la Paradoja de Simpson en estaciones de recarga
    new_cols['Charging_Home_x_CanCharge'] = (combined['Charging_Stations_Near_Home'] * home_charge_num).astype('float32')
    new_cols['Charging_Home_x_NoCharge'] = (combined['Charging_Stations_Near_Home'] * (1.0 - home_charge_num)).astype('float32')
    new_cols['Charging_Work_x_NoCharge'] = (combined['Charging_Stations_Near_Work'] * (1.0 - home_charge_num)).astype('float32')
    new_cols['Charging_Work_Home_Ratio'] = (
        (combined['Charging_Stations_Near_Work'] + 1.0) / (combined['Charging_Stations_Near_Home'] + 1.0)
    ).astype('float32')
    
    # 3. Regímenes Críticos y Zonas de Quiebre Estadístico
    print("[*] Añadiendo banderas de régimen (Millionaire Cliff, Dead Zone, 30k Spike)...")
    new_cols['is_30k_spike'] = (combined['Annual_Income_USD'] == 30000.0).astype('int8')
    new_cols['is_millionaire_cliff'] = (combined['Annual_Income_USD'] >= 170537.0).astype('int8')
    new_cols['is_dead_zone'] = (
        (combined['Annual_Income_USD'] >= 38000.0) & (combined['Annual_Income_USD'] <= 42000.0)
    ).astype('int8')
    new_cols['is_env_hater'] = (combined['Environmental_Concern_Level'] == 1.0).astype('int8')
    
    # 4. Markus Smooth Keys (como enteros eficientes)
    new_cols['income_100_int'] = np.floor(combined['Annual_Income_USD'] / 100.0).astype('int32')
    new_cols['income_1000_int'] = np.floor(combined['Annual_Income_USD'] / 1000.0).astype('int32')
    new_cols['commute_int'] = np.floor(combined['Daily_Commute_km']).astype('int16')
    
    # 5. Descomposición de Dígitos Sintéticos
    print("[*] Extrayendo dígitos sintéticos (posiciones 10^-4 hasta 10^3)...")
    digit_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
    for c in digit_cols:
        series_clean = combined[c].fillna(0).values
        for k in range(-4, 4):
            new_cols[f"{c}_digit{k}"] = ((series_clean // (10**k)) % 10).astype('int8')
            
    # 6. Mapeo de Medias Estadísticas del Dataset Original
    print("[*] Cruzando medias estadísticas del Dataset Original de Kaggle...")
    orig_global_mean = float(orig[TARGET].mean()) if TARGET in orig.columns else 0.1746
    raw_cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    for col in raw_cat_cols + digit_cols:
        if col in orig.columns:
            stats_map = orig.groupby(col, observed=False)[TARGET].mean().to_dict()
            new_cols[f"{col}_org_mean"] = combined[col].map(stats_map).fillna(orig_global_mean).astype('float32')
            
    # 7. Global Frequency Encoding
    print("[*] Calculando Frequency Encoding global...")
    fe_candidates = raw_cat_cols + ['income_100_int', 'income_1000_int', 'commute_int'] + [f"{c}_digit{k}" for c in digit_cols for k in range(-2, 2)]
    for col in fe_candidates:
        if col in combined.columns:
            freq = combined[col].value_counts(normalize=True).to_dict()
            new_cols[f"{col}_fe"] = combined[col].map(freq).fillna(0.0).astype('float32')
        elif col in new_cols:
            s = pd.Series(new_cols[col])
            freq = s.value_counts(normalize=True).to_dict()
            new_cols[f"{col}_fe"] = s.map(freq).fillna(0.0).astype('float32')
            
    # Codificar variables categóricas originales como category codes
    for col in raw_cat_cols:
        combined[col] = combined[col].astype('category').cat.codes.astype('int16')
        
    # Unir todo en una sola operación sin fragmentación
    df_new = pd.DataFrame(new_cols)
    combined = pd.concat([combined, df_new], axis=1)
    del new_cols, df_new
    gc.collect()
    
    # Separar train y test
    X_train = combined.iloc[:n_train].drop(columns=['id', 'Number_of_Cars_Owned'], errors='ignore')
    X_test = combined.iloc[n_train:].drop(columns=['id', 'Number_of_Cars_Owned'], errors='ignore')
    
    # Definir columnas para Target Encoding dentro de los Folds
    target_encode_cols = raw_cat_cols + ['income_100_int', 'income_1000_int', 'commute_int', 'Age', 'Environmental_Concern_Level']
    target_encode_cols = [c for c in target_encode_cols if c in X_train.columns]
    
    print(f"[+] Matriz final construida exitosamente:")
    print(f"    - Total Features: {X_train.shape[1]}")
    print(f"    - Filas Train: {len(X_train):,} | Filas Test: {len(X_test):,}")
    print(f"    - Columnas para Target Encoding: {len(target_encode_cols)}")
    
    return X_train, y_train, X_test, test_ids, target_encode_cols

if __name__ == "__main__":
    X_tr, y_tr, X_te, ids, te_cols = build_grandmaster_features()
    print("Features primeras 10:", list(X_tr.columns[:10]))
    print("Features últimas 10:", list(X_tr.columns[-10:]))
