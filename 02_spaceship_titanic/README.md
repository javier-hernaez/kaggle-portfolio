# 🚀 Spaceship Titanic

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.80500-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Rank](https://img.shields.io/badge/Leaderboard%20Rank-%23473-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://python.org)

Solución competitiva para el reto **Spaceship Titanic** de Kaggle utilizando un ensemble de **Gradient Boosting (LightGBM + CatBoost)** con ingeniería de características avanzada de dominio espacial.

---

## 📊 Resumen de Resultados

| Modelo | Estrategia | CV (5 Folds) | Kaggle Public Score |
| :--- | :--- | :---: | :---: |
| **LightGBM** | Hiperparámetros ajustados (max_depth=5, lr=0.03) | 0.8095 | - |
| **CatBoost** | Árboles simétricos (depth=5, lr=0.03) | 0.8105 | - |
| **Ensemble (LGBM + CatBoost)** | **Votación suave (Soft Voting)** | **0.8110** | **0.80500** 🏆 |

---

## 🧠 Ingeniería de Características Aplicada

1. **Lógica cruzada de Criocongelación (`CryoSleep`)**:
   * Si un pasajero estaba en criosueño, sus gastos en servicios (`RoomService`, `FoodCourt`, `Spa`, etc.) fueron estrictamente **`0`**.
   * Si un pasajero tuvo gastos superiores a 0, con total certeza **NO estaba en criosueño** (`CryoSleep = False`), resolviendo cientos de valores nulos de forma exacta.
2. **Desglose de Cabina (`Deck / Num / Side`)**:
   * `Deck` (Cubierta de la A a la T) y `Side` (`P` para babor / `S` para estribor).
   * **Correlación Cubierta - Planeta**: Las cubiertas `A, B, C, T` pertenecen exclusivamente a pasajeros de **Europa**; la cubierta `G` pertenece a pasajeros de la **Tierra**.
3. **Agrupación de Pasajeros (`PassengerId` format: `gggg_pp`)**:
   * Se extrajo el `GroupId` y el tamaño del grupo de viaje (`GroupSize`), identificando quiénes viajaban solos y quiénes en grupo.
4. **Transformación Logarítmica de Gastos**:
   * Aplicación de `log1p` para mitigar la asimetría de los pasajeros que gastaron miles de créditos frente a los que gastaron cero.

---

## 📂 Estructura de la Carpeta

```text
02_spaceship_titanic/
├── README.md               # Este documento
├── data/                   # train.csv, test.csv, sample_submission.csv
├── src/
│   └── model.py            # Script completo de preprocesamiento, CV y predicción
├── submissions/            # Histórico de envíos generados
└── submission.csv          # Última sumisión enviada a Kaggle (Score: 0.80500)
```

---

## 🚀 Cómo Reproducir los Resultados

```powershell
python 02_spaceship_titanic/src/model.py
kaggle competitions submit -c spaceship-titanic -f 02_spaceship_titanic/submission.csv -m "Ensemble LightGBM + CatBoost"
```
