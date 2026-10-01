# 🚀 Spaceship Titanic

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.80780-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Rank](https://img.shields.io/badge/Leaderboard%20Rank-%23248%20%2F%201589-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva de alto rendimiento para el reto **Spaceship Titanic** de Kaggle utilizando un ensemble tri-motor **(CatBoost + LightGBM + XGBoost)** con validación estratificada de 10 folds, calibración out-of-fold y 42 variables de ingeniería de características.

---

## 📊 Resumen de Resultados

| Modelo | Estrategia | CV (10-Fold OOF) | OOF ROC-AUC | Kaggle Public Score |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM** | Gradient Boosting estándar (max_depth=6, 350 trees) | 0.81399 | 0.90389 | - |
| **XGBoost** | Histogram-based trees (max_depth=5, 300 trees) | 0.81295 | 0.90400 | - |
| **CatBoost** | Árboles simétricos (depth=6, 700 iters) | 0.81652 | 0.90613 | - |
| **Tri-Engine Blend** | **Ensemble ponderado (0.50 CB + 0.30 XGB + 0.20 LGB)** | **0.81652** | **0.90635** | **`0.80780`** 🏆 (**#248**) |

---

## 🧠 Ingeniería de Características Aplicada (42 Variables)

1. **Imputación Semántica Cruzada por Grupo (`GroupId`) y Familia (`LastName`)**:
   * Los pasajeros del mismo `GroupId` (extraído del `PassengerId` `gggg_pp`) y del mismo apellido (`LastName`) comparten de forma determinista `HomePlanet`, `Destination`, `Deck` y `Side`.
   * Restricción VIP: Pasajeros de la Tierra nunca son VIP (`VIP = False`).
   * Pasajeros en `CryoSleep` o menores de 13 años no pueden tener gastos en amenidades (se fijan rígidamente a `0`).
   * Pasajeros con gastos > 0 se infieren deterministamente como `CryoSleep = False`.

2. **Geometría y Posicionamiento en la Nave**:
   * Desglose de `Cabin`: `Deck`, `CabinNum`, `Side` (`P` para babor / `S` para estribor).
   * **`CabinPosInDeck`**: Posición normalizada de la cabina respecto al número máximo de cabina dentro de cada cubierta específica (escala continua 0 a 1 de proa a popa).
   * **`CabinRegion`**: Sector longitudinal de la nave mediante discretización (`CabinNum // 300`).
   * **`DeckSide`**: Interacción unificada cubierta-costado.

3. **Desglose y Estructura del Gasto**:
   * División temática: `LuxurySpend` (`Spa` + `VRDeck` + `RoomService`) vs `EssentialSpend` (`FoodCourt` + `ShoppingMall`).
   * `AmenitiesCount`: Conteo de servicios distintos consumidos.
   * `col_pct`: Porcentaje de gasto relativo asignado a cada amenidad sobre el gasto total.
   * Flags binarios por servicio (`Has_RoomService`, etc.) y razón de gasto por edad (`SpendPerAge`).

4. **Agregaciones Sociales y de Grupo**:
   * Métricas a nivel grupo: `GroupTotalExpense`, `GroupMeanExpense` y `GroupExpenseRatio` (cuota del gasto familiar atribuida al pasajero).
   * `GroupCryoCount` y `GroupCryoRate`: Proporción del grupo familiar que viajaba en criocongelación.

---

## 📂 Estructura de la Carpeta

```text
02_spaceship_titanic/
├── README.md               # Este documento
├── data/                   # train.csv, test.csv, sample_submission.csv
├── src/
│   └── model.py            # Pipeline SOTA: Preprocesamiento, 10-Fold CV, Tri-Engine & Envíos
├── submissions/            # Histórico de envíos generados (incluye submission_tri_engine_10f.csv)
└── submission.csv          # Última sumisión enviada a Kaggle (Score: 0.80780)
```

---

## 🚀 Cómo Reproducir los Resultados

```powershell
python 02_spaceship_titanic/src/model.py
kaggle competitions submit -c spaceship-titanic -f 02_spaceship_titanic/submission.csv -m "SOTA Tri-Engine Ensemble 10-Fold Bagging (CatBoost + LightGBM + XGBoost)"
```
