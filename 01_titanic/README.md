# 🚢 Titanic - Machine Learning from Disaster

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.81578-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/titanic)
[![Rank](https://img.shields.io/badge/Leaderboard%20Rank-%23246%20(Top%202.5%25)-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/titanic)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva para el reto clásico **Titanic: Machine Learning from Disaster** de Kaggle, alcanzando una puntuación pública legítima de **0.81578** (Top 2.5% de más de 9.600 equipos) sin recurrir a fugas de datos históricos (*data leakage*).

---

## 📊 Resumen de Resultados y Progresión

| Modelo / Enfoque | Estrategia | CV (5 Folds) | Kaggle Public Score |
| :--- | :--- | :---: | :---: |
| **1. Gender Baseline** | Toda mujer vive, todo hombre muere | 0.786 | 0.76555 |
| **2. Baseline Random Forest** | Títulos, tamaño familiar e imputación simple | 0.8305 | 0.77751 |
| **3. Advanced Ensemble** | CatBoost + LightGBM + Random Forest con Soft Voting | 0.8429 | 0.77751 |
| **4. Pure WCG** | Reglas de grupos y familias de Chris Deotte | - | 0.80143 |
| **5. Champion WCG Model** | **Agrupación de parentesco, billetes y cabinas contiguas** | **0.8440** | **0.81578** 🏆 |

---

## 🧠 La Estrategia Clave: Woman-Child-Group (WCG)

En el naufragio del Titanic, el protocolo fue *"mujeres y niños primero"*. Sin embargo, **las familias y grupos de viaje actuaban como una unidad**:

1. **Familias que no llegaron a los botes**: Si una familia de 3ª clase quedó atrapada y los registros de entrenamiento muestran que sus miembros fallecieron (como las familias *Goodwin, Sage, Lefebre*), **todas las mujeres y niños de esa familia fallecieron** (rompiendo la regla general de que las mujeres sobreviven).
2. **Familias que lograron evacuar**: Si un grupo familiar consiguió subir a un bote salvavidas, los **niños varones (`Master`) de esa familia sobrevivieron** (rompiendo la regla general de que los varones mueren).
3. **Agrupación por Billetes (`Ticket`)**: Se conectaron grupos que viajaban juntos (sirvientes, amigos, parejas) compartiendo billetes o números contiguos (`Ticket[:-1] + 'X'`).

---

## 📂 Estructura de la Carpeta

```text
01_titanic/
├── README.md               # Este documento
├── download_data.py        # Descarga de datos vía Kagglehub / GitHub fallback
├── data/                   # Archivos oficiales de la competición
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── eda.py              # Script de análisis exploratorio
├── src/
│   ├── baseline_model.py   # Random Forest baseline
│   ├── advanced_model.py   # Ensemble CatBoost + LightGBM + RF
│   └── wcg_model.py        # Modelo campeón WCG (0.81578)
├── submissions/            # Histórico de predicciones generadas
└── submission.csv          # Última sumisión enviada a Kaggle
```

---

## 🚀 Cómo Reproducir los Resultados

### 1. Activar el entorno virtual
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Entrenar y generar la predicción campeona
```powershell
python 01_titanic/src/wcg_model.py
```

### 3. Enviar a Kaggle mediante la CLI
```powershell
kaggle competitions submit -c titanic -f 01_titanic/submission.csv -m "Champion WCG Model"
```
