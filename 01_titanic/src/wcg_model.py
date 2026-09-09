"""
Modelo Campeón Legítimo para Titanic (Kaggle)
Implementación completa de la técnica Woman-Child-Group (WCG) de Chris Deotte.
Alcanza una puntuación pública de 0.81578 en Kaggle.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUB_DIR = os.path.join(os.path.dirname(__file__), "..", "submissions")

def run_wcg_model():
    print("=" * 60)
    print(" ENTRENAMIENTO DEL MODELO WOMAN-CHILD-GROUP (WCG)")
    print("=" * 60)
    
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    data = pd.concat([train, test], sort=False).reset_index(drop=True)
    
    # 1. Segmentación natural de pasajeros
    data['PersonType'] = 'man'
    data.loc[data['Name'].str.contains('Master', na=False), 'PersonType'] = 'boy'
    data.loc[data['Sex'] == 'female', 'PersonType'] = 'woman'
    
    # 2. Extracción de apellidos y ticket modificado (agrupación de cabinas contiguas)
    data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0].strip())
    data['TicketMod'] = data['Ticket'].apply(lambda x: x[:-1] + 'X' if len(x) > 1 else x)
    
    # 3. Construcción del identificador de grupo (Familia)
    data['GroupId'] = (
        data['Surname'] + '-' + 
        data['Pclass'].astype(str) + '-' + 
        data['TicketMod'] + '-' + 
        data['Fare'].round(2).astype(str) + '-' + 
        data['Embarked'].fillna('S')
    )
    # Los hombres adultos no forman parte del núcleo Woman-Child Group
    data.loc[data['PersonType'] == 'man', 'GroupId'] = 'noGroup'
    
    # Caso histórico especial documentado: hermanas que viajaban juntas con distinto apellido
    # Mrs Wilkes (Needs) es hermana de Mrs Hocking (Needs)
    idx_893 = data[data['PassengerId'] == 893].index[0]
    idx_775 = data[data['PassengerId'] == 775].index[0]
    data.loc[idx_893, 'GroupId'] = data.loc[idx_775, 'GroupId']
    
    # Filtrar grupos de tamaño 1 (personas que viajaban solas)
    group_counts = data['GroupId'].value_counts()
    small_groups = group_counts[group_counts <= 1].index
    data.loc[data['GroupId'].isin(small_groups), 'GroupId'] = 'noGroup'
    
    # 4. Conectar nanas y familiares que comparten billete
    data['TicketId'] = (
        data['Pclass'].astype(str) + '-' + 
        data['TicketMod'] + '-' + 
        data['Fare'].round(2).astype(str) + '-' + 
        data['Embarked'].fillna('S')
    )
    
    ticket_to_group = data[data['GroupId'] != 'noGroup'].set_index('TicketId')['GroupId'].to_dict()
    for idx in data[(data['PersonType'] != 'man') & (data['GroupId'] == 'noGroup')].index:
        tid = data.loc[idx, 'TicketId']
        if tid in ticket_to_group:
            data.loc[idx, 'GroupId'] = ticket_to_group[tid]
            
    # 5. Calcular tasa de supervivencia del grupo en Train
    train_data = data[data['PassengerId'] <= 891]
    known_groups = train_data[train_data['GroupId'] != 'noGroup'].groupby('GroupId')['Survived'].mean()
    data['GroupSurvival'] = data['GroupId'].map(known_groups)
    
    # Para grupos presentes solo en test:
    # 3ª clase suele perecer (0.0), 1ª y 2ª clase suele sobrevivir (1.0)
    unknown_groups = data[(data['GroupId'] != 'noGroup') & (data['GroupSurvival'].isna())]
    for idx in unknown_groups.index:
        if data.loc[idx, 'Pclass'] == 3:
            data.loc[idx, 'GroupSurvival'] = 0.0
        else:
            data.loc[idx, 'GroupSurvival'] = 1.0
            
    # 6. Reglas de Decisión WCG
    data['Predict'] = 0
    # Regla base: Mujeres sobreviven, hombres mueren
    data.loc[data['Sex'] == 'female', 'Predict'] = 1
    
    # Excepción 1: Mujeres en familias donde todos murieron -> Perecen (0)
    females_die = (data['PersonType'] == 'woman') & (data['GroupSurvival'] == 0.0)
    data.loc[females_die, 'Predict'] = 0
    
    # Excepción 2: Niños varones (Master) en familias donde todos vivieron -> Sobreviven (1)
    boys_live = (data['PersonType'] == 'boy') & (data['GroupSurvival'] == 1.0)
    data.loc[boys_live, 'Predict'] = 1
    
    test_rows = data[data['PassengerId'] > 891]
    
    # 7. Guardar predicciones
    sub_df = test_rows[['PassengerId', 'Predict']].rename(columns={'Predict': 'Survived'})
    out_file = os.path.join(SUB_DIR, "submission_champion_wcg.csv")
    root_file = os.path.join(os.path.dirname(__file__), "..", "submission.csv")
    
    sub_df.to_csv(out_file, index=False)
    sub_df.to_csv(root_file, index=False)
    
    print(f"[+] Predicciones generadas con éxito:")
    print(f"    - {out_file}")
    print(f"    - {root_file}")
    print(f"\nDistribución: {sub_df['Survived'].value_counts().to_dict()} (Supervivencia: {sub_df['Survived'].mean():.1%})")

if __name__ == "__main__":
    run_wcg_model()
