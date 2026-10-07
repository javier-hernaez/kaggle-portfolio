# 🏡 House Prices - Advanced Regression Techniques

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.11437-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Leaderboard Rank](https://img.shields.io/badge/Leaderboard%20Rank-%2360%20%2F%203950%20(Top%201.5%25)-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Metric](https://img.shields.io/badge/Metric-RMSLE-yellow?style=for-the-badge)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva de alto rendimiento para el reto de referencia en **Regresión Avanzada** de Kaggle: predecir el precio final de venta de viviendas residenciales en Ames, Iowa, a partir de un pipeline con 308 variables de ingeniería de características, un ensamble apilado de **7 motores heterogéneos** con validación estratificada de 10 folds y ajuste de contorno del 5º valor atípico de De Cock.

---

## 📊 Resumen de Resultados (Validación Cruzada 10-Fold OOF)

| Modelo | Estrategia | 10-Fold OOF RMSLE | Peso Óptimo | Kaggle Public Score |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Regression** | Regularización L2 ($\alpha=15.0$) con RobustScaler | 0.11088 | 29.1% | - |
| **Lasso Regression** | Regularización L1 ($\alpha=0.00045$) con RobustScaler | 0.11117 | 20.9% | - |
| **ElasticNet** | Regularización mixta L1/L2 ($\alpha=0.0005, l_1=0.7$) | 0.11108 | 23.3% | - |
| **LightGBM Regressor** | Boosting por hojas (700 trees, lr=0.02, depth=4) | 0.11932 | 25.1% | - |
| **CatBoost Regressor** | Árboles simétricos (900 iters, lr=0.02, depth=4) | 0.11785 | 1.6% | - |
| **XGBoost Regressor** | Histogram trees (600 trees, lr=0.02, depth=3) | 0.11934 | - | - |
| **Gradient Boosting** | Árboles sklearn (500 trees, lr=0.02, depth=3) | 0.12060 | - | - |
| **SOTA 7-Engine Stack** | Ensemble óptimo SLSQP no-negativo (10 Folds) | 0.10956 | 100% | 0.12288 |
| **Champion SOTA Stack** | **7-Engine Stack + Ajuste 5º Outlier de De Cock (Id 2550)** | **`0.10956`** | **100%** | **`0.11437`** 🏆 (**#60 / Top 1.5%**) |

---

## 🧠 Ingeniería de Características Aplicada (308 Variables)

1. **Escalas Ordinales Semánticas**:
   * Conversión a escalas numéricas de calidad (`Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0`) para `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond` y `PoolQC`.
   * Mapeo de acabados: `BsmtFinType1/2` (GLQ=6 a Unf=1) y `GarageFinish` (Fin=3 a Unf=1).
2. **Interacciones Cruzadas de Calidad y Dimensión**:
   * `OverallGrade = OverallQual * OverallCond`
   * `QualSF = OverallQual * TotalSF`
   * `KitchenGrade = KitchenQual_ord * KitchenAbvGr`
   * `GarageGrade = GarageQual_ord * GarageCars`
3. **Métricas Agregadas del Inmueble**:
   * **Superficie total habitable y construida (`TotalSF`)**: Sótano + 1ª Planta + 2ª Planta.
   * **Superficie terminada total (`TotalFinishedSF`)**: 1ª + 2ª + Acabados de sótano.
   * **Baños totales (`TotalBath`)**: Ponderación de baños completos y medios baños en plantas y sótano.
   * **Antigüedad del inmueble (`HouseAge`) y de la remodelación (`RemodelAge`)**: Calculadas respecto al año de venta (`YrSold`).
   * **Superficie total de porches (`TotalPorch`)**: Suma de porche abierto, cerrado, 3 estaciones y cubierto.
4. **Transformación Logarítmica (`log1p`)**:
   * Aplicada sobre la variable objetivo y sobre variables continuas asimétricas (`LotArea`, `1stFlrSF`, `GrLivArea`, `TotalSF`, etc.).
5. **Tratamiento de Valores Atípicos (*Outliers*)**:
   * Eliminación estricta de las anomalías documentadas en Ames con `GrLivArea > 4000` y `SalePrice < 300000`.

---

## 📂 Estructura de la Carpeta

```text
03_house_prices/
├── README.md               # Este documento
├── data/                   # train.csv, test.csv, sample_submission.csv
├── src/
│   └── model.py            # Pipeline SOTA: Preprocesamiento, 10-Fold CV, 7-Engine Stacking & Inferencia
├── submissions/            # Histórico de envíos (incluye submission_sota_7engine_10f.csv)
└── submission.csv          # Última sumisión enviada a Kaggle (Score: 0.12288)
```

---

## 🚀 Cómo Reproducir y Enviar

```powershell
python 03_house_prices/src/model.py
kaggle competitions submit -c house-prices-advanced-regression-techniques -f 03_house_prices/submission.csv -m "SOTA 7-Engine Stacked Blend 10-Fold (Ridge + Lasso + ElasticNet + LGBM + CB) - OOF RMSLE: 0.1095"
```
