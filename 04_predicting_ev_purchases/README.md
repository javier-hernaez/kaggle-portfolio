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
* **Modelos Base:**
  * **LightGBM Classifier:** 400 estimadores, profundidad 6, submuestreo de características y filas.
  * **CatBoost Classifier:** 400 iteraciones con optimización para ROC-AUC.
  * **XGBoost Classifier:** 400 árboles con regularización y early stopping.
* **Ensamble:** Fusión ponderada de probabilidades Out-Of-Fold (OOF) y promediado de predicciones sobre el conjunto de test (test fold bagging).

### 🏆 Resultados de Validación Cruzada (OOF ROC-AUC)

| Modelo | Métrica | OOF ROC-AUC | Peso en Ensamble |
| :--- | :---: | :---: | :---: |
| **LightGBM** | ROC-AUC | `0.941027` | 0.10 |
| **CatBoost** | ROC-AUC | `0.941551` | 0.30 |
| **XGBoost** | ROC-AUC | `0.941768` | 0.60 |
| **Ensemble (Blend OOF)** | **ROC-AUC** | **`0.941828`** | **1.00** |

*Rendimiento por Fold del Ensamble:*
* Fold 1: `0.940423`
* Fold 2: `0.941489`
* Fold 3: `0.942590`
* Fold 4: `0.942127`
* Fold 5: `0.941771`

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
