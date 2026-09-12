# 📢 Kaggle NLP with Disaster Tweets

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Disaster_Tweets-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nlp-getting-started)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Transformers-blueviolet?style=for-the-badge)](https://huggingface.co/)

Solución de Procesamiento de Lenguaje Natural (NLP) para la competición **Natural Language Processing with Disaster Tweets**.

El objetivo del reto consiste en predecir si un tweet hace referencia a una catástrofe o emergencia real (`target = 1`) o si utiliza vocabulario de alerta de forma metafórica o coloquial (`target = 0`), evaluado mediante el **F1-Score**.

---

## 📊 Hallazgos Clave del Análisis Exploratorio (EDA)

El conjunto de datos cuenta con **7.613 tweets de entrenamiento** y **3.263 de prueba**.

* **Distribución de Clases:**
  * No Desastre (`0`): **57.03%** (4.342 tweets).
  * Desastre Real (`1`): **42.97%** (3.271 tweets).
* **Valores Faltantes:**
  * `text`: 0% nulos.
  * `keyword`: 0.80% nulos (con frecuencia codificado en formato URL como `oil%20spill`).
  * `location`: 33.27% nulos (datos muy ruidosos y geolocalizaciones subjetivas).
* **Longitud y Estilometría:**
  * Los tweets de catástrofes reales son sensiblemente más largos (media de **108.1 caracteres** vs **95.7 caracteres** en no desastres).
  * Palabras clave letales (100% asociadas a desastre): `derailment`, `debris`, `wreckage`, `typhoon`, `outbreak`.
  * Palabras clave metafóricas/ruido (<5% desastre): `screaming`, `blew up`, `panicking`, `body bags`, `aftershock`.

---

## 🛠️ Arquitectura del Pipeline de NLP

```text
05_disaster_tweets/
├── README.md                          # Documentación del proyecto
├── download_data.py                   # Descarga de datos vía Kaggle API
├── submission.csv                     # Submission activa
├── data/                              # train.csv, test.csv, sample_submission.csv
├── submissions/                       # Historial de predicciones generadas
└── src/
    ├── eda.py                         # Análisis exploratorio y patrones estadísticos
    ├── text_cleaner.py                # Limpieza (HTML unescape, mojibake, URLs, menciones, contracciones)
    ├── features.py                    # TF-IDF (palabras + caracteres) y meta-features
    ├── evaluate.py                    # Optimización de umbral de corte para F1
    ├── baseline_model.py              # 5-Fold Stratified CV (Logistic Regression + Calibrated Ridge)
    └── train_transformer_kaggle.py    # Pipeline DeBERTa-v3 listo para GPU en Kaggle/Colab
```

### 1. Limpieza de Texto (`text_cleaner.py`)
* **Corrección de Mojibake y Entidades HTML:** Decodificación de caracteres corruptos de Twitter (`\x89Û_`, `\x89Ûª`, `&amp;` $\to$ `and`).
* **Preservación de Hashtags:** Reemplazo de `#wildfire` por la palabra `wildfire` para aprovechar la semántica en el vocabulario.
* **Normalización de URLs y Menciones:** Sustitución por tokens normalizados (`url`, `mention`).
* **Expansión de contracciones:** Normalización de formas cortas en inglés (`can't` $\to$ `cannot`, `won't` $\to$ `will not`).

### 2. Extracción de Características (`features.py`)
* **TF-IDF Palabras:** N-gramas (1, 2) con escala `sublinear_tf=True` (hasta 15.000 términos).
* **TF-IDF Caracteres:** N-gramas (3, 5) a nivel de carácter para captar raíces léxicas y erratas (hasta 20.000 términos).
* **Meta-Features Estilométricas:** Recuento de caracteres, palabras, ratio de mayúsculas (gritos de alerta), signos de exclamación y de interrogación, número de enlaces y coincidencia de la palabra clave en el cuerpo del tweet.

### 3. Modelado y Ensamble (`baseline_model.py`)
* **Validación Cruzada:** 5-Fold Stratified K-Fold con semilla fija (`seed=42`).
* **Modelos:**
  1. *Logistic Regression* con regularización $L_2$ y solver lineal.
  2. *Calibrated Ridge Classifier* con calibración de probabilidad sigmoidea.
  3. *Ensemble Blend:* Promedio equiponderado de probabilidades.
* **Threshold Tuning (`evaluate.py`):** Optimización del umbral de decisión sobre las predicciones Out-of-Fold (OOF) para maximizar directamente el F1-Score binario.

---

## 📈 Resultados de Validación Cruzada (5-Fold CV)

| Modelo | ROC-AUC | F1-Score (th = 0.50) | Umbral Óptimo | F1-Score Óptimo | Precisión | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | `0.87278` | `0.76471` | `0.435` | **`0.77196`** | 0.77979 | 0.76429 |
| **Calibrated Ridge Classifier** | `0.87043` | `0.76423` | `0.465` | **`0.76874`** | 0.80360 | 0.73678 |
| **Blend Ensemble (LR + Ridge)** | **`0.87305`** | `0.76569` | `0.450` | **`0.77048`** | 0.79017 | 0.75176 |

> [!TIP]
> **Ganancia por Optimización de Umbral:**
> Al desplazar el punto de corte de `0.50` a `0.435 - 0.450`, el modelo compensa el coste asimétrico de los falsos negativos en la métrica F1, logrando una mejora de **+0.007 F1** sin requerir modelos más pesados.

---

## 🚀 Próximo Nivel: Transformers con GPU (`train_transformer_kaggle.py`)

Para superar la barrera de $\text{F1} > 0.83$, el repositorio incluye el script `src/train_transformer_kaggle.py`, diseñado para ejecutarse directamente en un entorno con GPU (Kaggle Notebooks o Google Colab):
* Arquitectura: `microsoft/deberta-v3-small` / `deberta-v3-base`.
* Tokenización contextual unificada: `"Keyword: {kw}. Tweet: {text}"`.
* Fine-Tuning con Hugging Face `Trainer` y evaluación periódica de F1.
