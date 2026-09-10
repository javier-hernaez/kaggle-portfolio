"""
Feature Engineering Pipeline for Playground Series s6e9:
Predicting Electric Vehicle Purchases
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr  # Fast standard normal CDF

def build_features(df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """
    Realiza la ingeniería de variables optimizada para predecir Will_Buy_EV con ROC-AUC.
    Aprovecha la función generadora identificada (probit latent DGP) e interacciones de alto valor.
    """
    data = df.copy()
    
    # Extraer o identificar target
    target = None
    if 'Will_Buy_EV' in data.columns:
        target = (data['Will_Buy_EV'] == 'Yes').astype(int)
        
    # 1. Variables binarias y ordinales
    data['Subsidy_Available_num'] = (data['Subsidy_Available'] == 'Yes').astype(float)
    data['Home_Charging_Possible_num'] = (data['Home_Charging_Possible'] == 'Yes').astype(float)
    
    anxiety_map = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
    data['Range_Anxiety_num'] = data['Range_Anxiety_Level'].map(anxiety_map).fillna(0.0)
    
    # 2. Indicadores específicos de Range Anxiety
    data['Med_Range_Anxiety'] = (data['Range_Anxiety_Level'] == 'Medium').astype(float)
    data['High_Range_Anxiety'] = (data['Range_Anxiety_Level'] == 'High').astype(float)
    
    # 3. Ingresos escalados y transformaciones
    data['Income_Scaled'] = data['Annual_Income_USD'] / 100000.0
    data['Log_Income'] = np.log1p(data['Annual_Income_USD'])
    data['Income_Per_Car'] = data['Annual_Income_USD'] / (data['Number_of_Cars_Owned'] + 1e-5)
    
    # 4. Aproximación Probit del Proceso Generador de Datos (DGP)
    # Buy Score = 1.2 * (Income/100k) + 0.6 * Concern + 2.0 * Subsidy - 1.0 * MedAnx - 3.0 * HighAnx - 5.5
    data['DGP_Probit_Score'] = (
        1.2 * data['Income_Scaled']
        + 0.6 * data['Environmental_Concern_Level']
        + 2.0 * data['Subsidy_Available_num']
        - 1.0 * data['Med_Range_Anxiety']
        - 3.0 * data['High_Range_Anxiety']
        - 5.5
    )
    # Probabilidad acumulada normal estándar (probit link)
    data['DGP_Probit_Prob'] = ndtr(data['DGP_Probit_Score'])
    
    # 5. Puntuación Logit ajustada
    data['Logit_Latent_Score'] = (
        2.553 * data['Income_Scaled']
        + 1.315 * data['Environmental_Concern_Level']
        + 4.542 * data['Subsidy_Available_num']
        - 1.606 * data['Med_Range_Anxiety']
        - 3.594 * data['High_Range_Anxiety']
        + 0.237 * data['Home_Charging_Possible_num']
        - 0.0045 * data['Daily_Commute_km']
        - 8.95
    )
    data['Logit_Prob'] = 1.0 / (1.0 + np.exp(-data['Logit_Latent_Score']))
    
    # 6. Interacciones críticas de negocio / adopción EV
    data['Income_x_Subsidy'] = data['Income_Scaled'] * data['Subsidy_Available_num']
    data['Concern_x_Subsidy'] = data['Environmental_Concern_Level'] * data['Subsidy_Available_num']
    data['Concern_x_Income'] = data['Environmental_Concern_Level'] * data['Income_Scaled']
    
    # 7. Infraestructura de recarga y estrés de rango
    data['Total_Charging_Stations'] = (
        data['Charging_Stations_Near_Home'] + data['Charging_Stations_Near_Work']
    )
    data['Charging_Infrastructure_Index'] = (
        data['Total_Charging_Stations'] + 3.0 * data['Home_Charging_Possible_num']
    )
    data['Commute_Charging_Stress'] = data['Daily_Commute_km'] / (data['Total_Charging_Stations'] + 1.0)
    data['Commute_x_Anxiety'] = data['Daily_Commute_km'] * (data['Range_Anxiety_num'] + 1.0)
    
    # 8. Variables demográficas y de vehículos
    data['Age_Commute_Interaction'] = (data['Age'] / 50.0) * (data['Daily_Commute_km'] / 30.0)
    
    # 9. Codificación categórica
    cat_columns = ['Gender', 'City_Type', 'Current_Car_Type']
    for col in cat_columns:
        data[col] = data[col].astype('category').cat.codes
        
    # Columnas a excluir del modelado (id y target original si existiera)
    exclude_cols = {'id', 'Will_Buy_EV', 'Range_Anxiety_Level', 'Subsidy_Available', 'Home_Charging_Possible'}
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    return data[feature_cols], target, feature_cols

if __name__ == "__main__":
    import os
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), nrows=1000)
    X, y, cols = build_features(train)
    print(f"Features construidas exitosamente ({len(cols)} variables):")
    print(cols)
    print("\nPrimeras filas:")
    print(X.head(3))
