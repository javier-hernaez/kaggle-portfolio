"""
Análisis Exploratorio de Datos (EDA) para NLP with Disaster Tweets.
Examina distribución de clases, campos faltantes, longitud de textos y keywords.
"""

import os
import sys
import pandas as pd
import numpy as np

# Asegurar codificación utf-8 en terminal Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def run_eda():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[!] Error: Faltan archivos en {DATA_DIR}. Ejecuta download_data.py primero.")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("=" * 65)
    print(" 1. INFORMACION GENERAL DEL DATASET")
    print("=" * 65)
    print(f"Dimensiones de Train : {train.shape[0]:,} filas, {train.shape[1]} columnas")
    print(f"Dimensiones de Test  : {test.shape[0]:,} filas, {test.shape[1]} columnas")
    print(f"Columnas en Train    : {list(train.columns)}")
    print(f"Columnas en Test     : {list(test.columns)}")

    print("\n" + "=" * 65)
    print(" 2. DISTRIBUCION DE LA VARIABLE OBJETIVO (target)")
    print("=" * 65)
    counts = train['target'].value_counts()
    pcts = train['target'].value_counts(normalize=True) * 100
    for cls in [0, 1]:
        label = "No Desastre (0)" if cls == 0 else "Desastre Real (1)"
        print(f" - {label:<18}: {counts[cls]:,} tweets ({pcts[cls]:.2f}%)")

    print("\n" + "=" * 65)
    print(" 3. VALORES NULOS / FALTANTES")
    print("=" * 65)
    print("--- Train ---")
    for col in train.columns:
        n_missing = train[col].isnull().sum()
        print(f" - {col:<12}: {n_missing:>5} nulos ({n_missing/len(train)*100:5.2f}%)")
    print("--- Test ---")
    for col in test.columns:
        n_missing = test[col].isnull().sum()
        print(f" - {col:<12}: {n_missing:>5} nulos ({n_missing/len(test)*100:5.2f}%)")

    print("\n" + "=" * 65)
    print(" 4. ANALISIS DE LONGITUD DE TEXTO")
    print("=" * 65)
    train['char_len'] = train['text'].str.len()
    train['word_len'] = train['text'].apply(lambda x: len(str(x).split()))
    
    print("Caracteres por tweet (Media / Mediana / Max):")
    print(f" - General        : {train['char_len'].mean():.1f} / {train['char_len'].median():.0f} / {train['char_len'].max()}")
    print(f" - Desastre (1)   : {train[train['target']==1]['char_len'].mean():.1f} / {train[train['target']==1]['char_len'].median():.0f} / {train[train['target']==1]['char_len'].max()}")
    print(f" - No Desastre (0): {train[train['target']==0]['char_len'].mean():.1f} / {train[train['target']==0]['char_len'].median():.0f} / {train[train['target']==0]['char_len'].max()}")

    print("\nPalabras por tweet (Media / Mediana / Max):")
    print(f" - Desastre (1)   : {train[train['target']==1]['word_len'].mean():.1f} / {train[train['target']==1]['word_len'].median():.0f} / {train[train['target']==1]['word_len'].max()}")
    print(f" - No Desastre (0): {train[train['target']==0]['word_len'].mean():.1f} / {train[train['target']==0]['word_len'].median():.0f} / {train[train['target']==0]['word_len'].max()}")

    print("\n" + "=" * 65)
    print(" 5. TOP KEYWORDS CON MAYOR TASA DE DESASTRE (min 15 tweets)")
    print("=" * 65)
    kw_stats = train.groupby('keyword').agg(
        total=('target', 'count'),
        disasters=('target', 'sum'),
        disaster_rate=('target', 'mean')
    ).query('total >= 15').sort_values('disaster_rate', ascending=False)

    print("Top 10 keywords más letales / predictivas de desastre:")
    for kw, row in kw_stats.head(10).iterrows():
        print(f" - {kw:<25}: {row['disaster_rate']*100:5.1f}% desastre ({int(row['disasters'])}/{int(row['total'])})")

    print("\nTop 10 keywords menos asociadas a desastre real (metáforas/ruido):")
    for kw, row in kw_stats.tail(10).iterrows():
        print(f" - {kw:<25}: {row['disaster_rate']*100:5.1f}% desastre ({int(row['disasters'])}/{int(row['total'])})")

    print("\n" + "=" * 65)
    print(" 6. EJEMPLOS DE TWEETS (Metáforas vs Reales)")
    print("=" * 65)
    print("[+] Desastre Real (target = 1):")
    for sample in train[train['target']==1]['text'].head(2):
        print(f"  > \"{sample.strip()}\"")
    print("\n[-] No Desastre (target = 0) pero con keywords alarmantes:")
    metaphor_sample = train[(train['target']==0) & (train['keyword'].isin(['wrecked', 'bomb', 'screaming']))]['text'].head(2)
    for sample in metaphor_sample:
        print(f"  > \"{sample.strip()}\"")

if __name__ == "__main__":
    run_eda()
