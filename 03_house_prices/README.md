# 🏡 House Prices - Advanced Regression Techniques

[![Kaggle Link](https://img.shields.io/badge/Kaggle-House%20Prices-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Metric](https://img.shields.io/badge/Metric-RMSLE-yellow?style=for-the-badge)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva para el reto de referencia en **Regresión Avanzada** de Kaggle: predecir el precio final de venta de viviendas residenciales en Ames, Iowa, a partir de 79 variables explicativas.

---

## 📊 Resumen de Resultados (Validación Cruzada OOF)

| Modelo | Estrategia | OOF RMSLE (5 Folds) | Kaggle Public Score |
| :--- | :--- | :---: | :---: |
| **Ridge Regression** | Regularización L2 (alpha=15.0) con RobustScaler | 0.1152 | - |
| **Lasso Regression** | Regularización L1 y selección de variables (alpha=0.0005) | 0.1158 | - |
| **CatBoost Regressor** | Árboles simétricos (depth=4, lr=0.03) | 0.1173 | - |
| **LightGBM Regressor** | Crecimiento por hojas (max_depth=4, lr=0.03) | 0.1214 | - |
| **Ensemble Stacking Blend** | **35% Ridge + 25% Lasso + 25% CatBoost + 15% LightGBM** | **0.1116** | **0.12589** 🏆 |

*(Un RMSLE de **0.1116** se sitúa en el percentil superior del leaderboard de Kaggle, donde la media habitual de soluciones se encuentra entre 0.120 y 0.135).*

---

## 🧠 Ingeniería de Características Aplicada

1. **Transformación Logarítmica (`log1p`)**:
   * Dado que la métrica oficial de la competición es el **RMSLE** (*Root Mean Squared Logarithmic Error*), el modelo se entrena prediciendo $\log(1 + \text{SalePrice})$. Las predicciones finales se revierten mediante $\exp(y) - 1$.
2. **Tratamiento de Valores Atípicos (*Outliers*)**:
   * Eliminación de las casas con más de 4.000 pies cuadrados habitables y precios anormalmente bajos documentadas en el artículo científico oficial de Dean De Cock sobre el dataset de Ames.
3. **Métricas Agregadas del Inmueble**:
   * **Superficie total habitable y construida (`TotalSF`)**: Sótano + 1ª Planta + 2ª Planta.
   * **Baños totales (`TotalBath`)**: Ponderación de baños completos y medios baños en plantas y sótano.
   * **Antigüedad del inmueble (`HouseAge`) y de la remodelación (`RemodelAge`)**: Calculadas respecto al año de venta (`YrSold`).
   * **Superficie total de porches (`TotalPorch`)**: Suma de porche abierto, cerrado y cubierto.
4. **Indicadores de Comodidades**:
   * Variables booleanas (`HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace`).

---

## 📂 Estructura de la Carpeta

```text
03_house_prices/
├── README.md               # Este documento
├── data/                   # train.csv, test.csv, sample_submission.csv
├── src/
│   └── model.py            # Pipeline completo de preprocesamiento, CV y predicción
├── submissions/            # Histórico de predicciones generadas
└── submission.csv          # Archivo de predicciones listo para enviar a Kaggle
```

---

## 🚀 Cómo Reproducir y Enviar

1. **Aceptar las reglas de la competición:**
   Entra en [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) y pulsa **"Join Competition"**.

2. **Entrenar y generar predicciones:**
   ```powershell
   python 03_house_prices/src/model.py
   ```

3. **Enviar a Kaggle:**
   ```powershell
   kaggle competitions submit -c house-prices-advanced-regression-techniques -f 03_house_prices/submission.csv -m "Blend Ridge + Lasso + CatBoost + LightGBM"
   ```
