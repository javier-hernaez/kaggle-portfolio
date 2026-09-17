# 🚗 Kaggle Playground Series s6e9: Predicting Electric Vehicle Purchases

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_s6e9-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-green?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Yandex-yellow?style=for-the-badge)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-DMLC-red?style=for-the-badge)](https://xgboost.readthedocs.io/)

Solución de Machine Learning de alto rendimiento para la competición **Playground Series Season 6 Episode 9: Predicting Electric Vehicle Purchases**.

El objetivo del desafío es predecir la probabilidad de que un consumidor potencial compre un Vehículo Eléctrico (`Will_Buy_EV`), evaluado mediante el **Área bajo la Curva ROC (ROC-AUC)**.

---

## 📊 Resumen del Dataset y Hallazgos Clave (EDA)

El conjunto de datos comprende **668.665 observaciones de entrenamiento** y **286.571 de test**, sin valores nulos en ninguna de las variables.

* **Target (`Will_Buy_EV`):** Desbalanceado, con un **17.46% de compras afirmativas (`Yes`)** frente a un **82.54% negativas (`No`)**.
* **Disponibilidad de Subsidios (`Subsidy_Available`):** Factor determinante en la adopción.
  * Con subsidio (`Yes`): **27.47%** de intención de compra.
  * Sin subsidio (`No`): apenas un **0.58%** de intención de compra.
* **Nivel de Preocupación Ambiental (`Environmental_Concern_Level`):** Correlación de Pearson de $+0.464$.
  * Nivel 1: $0.56\%$ tasa de compra.
  * Nivel 5: $51.83\%$ tasa de compra.
* **Ansiedad de Autonomía (`Range_Anxiety_Level`):**
  * Baja: $18.90\%$
  * Media: $4.17\%$
  * Alta: $0.14\%$
* **Ingresos Anuales (`Annual_Income_USD`):** Correlación positiva de $+0.226$.

---

## 🔬 Descubrimiento del Proceso Generador de Datos (DGP)

A través del análisis de regresión y comportamiento sintético de los datos, se identificó que las etiquetas siguen un modelo latente de tipo **Probit**:

$$\text{Score} = 1.2 \times \frac{\text{Income}}{100,000} + 0.6 \times \text{Concern} + 2.0 \times \text{Subsidy} - 1.0 \times \text{MedAnxiety} - 3.0 \times \text{HighAnxiety} - 5.5$$

$$P(\text{Will\_Buy\_EV} = 1) = \Phi(\text{Score})$$

Donde $\Phi(\cdot)$ representa la función de distribución acumulada de la normal estándar. Este cálculo por sí solo alcanza un ROC-AUC de **0.9377**. El pipeline implementado expande este núcleo con interacciones de infraestructura, estrés de recarga y modelos no lineales en árbol.

---

## 🧠 Arquitectura de Ingeniería de Variables (`features.py`)

1. **Indicadores de DGP y Enlaces de Probabilidad:**
   * `DGP_Probit_Score` y `DGP_Probit_Prob` ($\Phi(Z)$).
   * `Logit_Latent_Score` y `Logit_Prob` vía regresión logística regularizada.
2. **Interacciones Económicas y de Subsidio:**
   * `Income_x_Subsidy`: Interacción multiplicativa de capacidad adquisitiva y ayuda estatal.
   * `Concern_x_Subsidy`: Sinergia entre conciencia ecológica y viabilidad económica.
   * `Income_Per_Car`: Ratio de ingresos distribuido entre el parque vehicular del hogar.
3. **Estrés de Autonomía e Infraestructura de Recarga:**
   * `Total_Charging_Stations`: Suma de cargadores en hogar y lugar de trabajo.
   * `Charging_Infrastructure_Index`: Ponderación combinada con capacidad de recarga domiciliaria.
   * `Commute_Charging_Stress`: Kilómetros diarios respecto a estaciones disponibles.
   * `Commute_x_Anxiety`: Distancia de traslado modulada por el nivel de ansiedad.

---

## ⚙️ Modelado y Validación Cruzada (`model.py`)

* **Estrategia de Validación:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True)`).
* **Ingeniería de Características Grandmaster (118 Features):**
  * Medias objetivo extraídas del dataset original de Kaggle (`EV_Adoption_and_Range_Anxiety_Dataset.csv`).
  * Descomposición de dígitos sintéticos en base 10 ($10^{-4}$ a $10^{3}$) para variables numéricas.
  * Zonas de quiebre estadístico: `is_millionaire_cliff` ($\ge 170.537$), `is_dead_zone` ($38k-42k$), `is_30k_spike`.
  * In-Fold Triple Target Encoding (`smooth='auto', 10, 100`) para 11 columnas clave.
* **Modelos de Ultra-Alta Resolución:**
  * **LightGBM Classifier:** `max_bin=1024`, `num_leaves=32`, `colsample_bytree=0.3`, `learning_rate=0.03`.
  * **XGBoost Classifier (Hist):** `max_bin=1024`, `max_depth=6`, `colsample_bytree=0.3`, `learning_rate=0.03`.
* **Ensamble:** Percentile Rank Averaging y test-fold bagging entre 10 modelos (5 folds $\times$ 2 modelos).

### 🏆 Resultados y Progresión en el Leaderboard de Kaggle

| Versión / Estrategia | OOF ROC-AUC | Score Público Kaggle | Posición Leaderboard | Percentil |
| :--- | :---: | :---: | :---: | :---: |
| **v1: Baseline Probit + GBDT** | `0.94183` | `0.94151` | #1.005 / 1.583 | Top 64% |
| **v2: Base Margin + Simpson Paradox** | `0.94203` | `0.94173` | #905 / 1.583 | Top 57% |
| **v3: Digit Decomp + High-Res Hist (118 feats)** | `0.94485` | `0.94470` | #589 / 1.583 | Top 37% |
| **v4: Nelder-Mead 10F XGB + Pure LGBM Super-Blend**| `0.94630` | `0.94642` | #266 / 2.216 | Top 12% |
| **v5: Meta-Stack Super-Ensemble (Lasso + Tie-Breaker)** | `0.94635` | `0.94650` | #88 / 2.216 | Top 4.0% |
| **⭐ v8: Multi-Anchor Consensus (Top-3 Anchors + 10F XGBoost)** | **`0.94640`** | **`0.94651`** | **#45 / 2.217** | **Top 2.0% (Percentil 98%)** |

*Detalle de la arquitectura campeona v8:*
* **Multi-Anchor Consensus Optimization:** Ponderación y agregación no lineal de los mejores envíos verificados del certamen (submission 56267408, 56267573, 56267624) junto al modelo XGBoost 10-fold entrenado con triple target encoding dinámico.
* **Ajustes de Frontera Discontinua:** Corrección matemática sobre las singularidades deterministas del generador sintético (cliff de ingresos $\ge \$170.537$, zona muerta de ingresos $\$31.004 - \$41.970$, corte de commute $\ge 83\text{km}$ y condición de spike en $\$30.000$).
* **Resolución Lexicográfica de Empates:** Cero empates en el conjunto de test gracias a desambiguación continua de alta resolución.

---

## 🚀 Instrucciones de Ejecución

1. **Descargar los datos:**
   ```bash
   python 04_predicting_ev_purchases/download_data.py
   ```

2. **Ejecutar Análisis Exploratorio:**
   ```bash
   python 04_predicting_ev_purchases/src/eda.py
   ```

3. **Entrenar Ensamble y Generar Predicciones:**
   ```bash
   python 04_predicting_ev_purchases/src/model.py
   ```

El archivo final listo para Kaggle se generará en:
`04_predicting_ev_purchases/submission.csv`
