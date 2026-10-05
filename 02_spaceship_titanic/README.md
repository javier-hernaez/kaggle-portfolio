# 🚀 Spaceship Titanic

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.80921-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Rank](https://img.shields.io/badge/Leaderboard%20Rank-%23135%20%2F%201574%20(Top%208.5%25)-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva de alto rendimiento para el reto **Spaceship Titanic** de Kaggle utilizando un ensemble tri-motor **(CatBoost + LightGBM + XGBoost)** con validación estratificada de 10 folds, calibración out-of-fold y 57 variables de ingeniería de características físicas y de comportamiento.

---

## 📊 Resumen de Resultados

| Modelo | Estrategia | CV (10-Fold OOF) | OOF ROC-AUC | Kaggle Public Score |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM** | Gradient Boosting estándar (max_depth=6, 400 trees) | 0.81847 | 0.90615 | - |
| **XGBoost** | Histogram-based trees (max_depth=5, 350 trees) | 0.81790 | 0.90601 | - |
| **CatBoost** | Árboles simétricos (depth=6, 750 iters) | 0.81571 | 0.90751 | - |
| **Tri-Engine Blend (v1)** | Ensemble ponderado (0.50 CB + 0.30 XGB + 0.20 LGB) | 0.81652 | 0.90635 | 0.80780 (#248) |
| **SOTA Grandmaster (v3)** | **Ensemble óptimo 57 características (Th: 0.485)** | **`0.81870`** | **`0.90751`** | **`0.80921`** 🏆 (**#135**) |

---

## 🧠 Ingeniería de Características Aplicada (57 Variables)

1. **Segmentación de Regímenes de Supervivencia (`PassengerType`)**:
   * `Child` (Edad < 13): 70% de supervivencia independiente de criocongelación.
   * `CryoAdult`: 83% de supervivencia global.
   * `SpenderAdult`: 30% de supervivencia global.
   * `ZeroSpenderAdult`: 53% de supervivencia.

2. **Regla de Oro de Criocongelación en Cubiertas Superiores (`IsCryo_HighDeck`)**:
   * Pasajeros adultos en `CryoSleep` ubicados en cubiertas `A, B, C, D, F` tienen una tasa de transporte determinista del **99.0%** (1.284 / 1.297).

3. **Disociación Asimétrica de Amenidades**:
   * `NegativeAmenities = RoomService + Spa + VRDeck` (fuerte correlación negativa, r = -0.55 en log).
   * `PositiveAmenities = FoodCourt + ShoppingMall` (correlación neutra/positiva).
   * `IsSpender_ZeroNeg`: Si un pasajero con gastos tiene `NegativeAmenities == 0`, su supervivencia sube al **64.4%**.

4. **Imputación Semántica Determinista por Grupo (`GroupId`) y Familia (`LastName`)**:
   * 100% de coherencia intragrupo en `HomePlanet` y `Side`.
   * Imputación de `CabinNum` por correlación espacial con `GroupId_int`.

5. **Geometría Espacial de la Nave**:
   * `CabinPosInDeck`: Posición relativa continua (0 a 1) de proa a popa dentro de cada cubierta.
   * `CabinRegion100` y `CabinRegion300`: Bloques longitudinales de cabinas.
   * Interacciones unificadas: `DeckSide`, `CryoDeck`, `CryoSide` y `CryoDeckSide`.

---

## 📂 Estructura de la Carpeta

```text
02_spaceship_titanic/
├── README.md               # Este documento
├── data/                   # train.csv, test.csv, sample_submission.csv
├── src/
│   └── model.py            # Pipeline SOTA: 57 Variables, 10-Fold CV, Tri-Engine & Inferencia
├── submissions/            # Histórico de envíos generados (incluye submission_57f_optimal_th485.csv)
└── submission.csv          # Última sumisión enviada a Kaggle (Score: 0.80921)
```

---

## 🚀 Cómo Reproducir los Resultados

```powershell
python 02_spaceship_titanic/src/model.py
kaggle competitions submit -c spaceship-titanic -f 02_spaceship_titanic/submission.csv -m "SOTA Tri-Engine (50% CB + 30% XGB + 20% LGB) with 57 Physics Features & Th 0.485"
```
