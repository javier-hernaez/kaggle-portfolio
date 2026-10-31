# 🤖 Kaggle: Name That LLM (Featured Code Competition)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/name-that-llm)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-green?style=for-the-badge)](https://lightgbm.readthedocs.io/)

Este proyecto contiene la arquitectura, ingeniería de variables y modelos para la competición destacada de Kaggle **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** ($50,000 USD de premio).

---

## 🎯 Objetivo y Desafío de la Competición

El reto consiste en realizar **atribución de autoría** (*authorship attribution*) para identificar qué gran modelo de lenguaje (entre **18 modelos de frontera** de 6 laboratorios líderes) generó un pasaje de texto no etiquetado de entre **40 y 250 palabras**.

### Modelos Objetivo (18 clases):
* **OpenAI:** `gpt-5.5`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-6-astra`
* **Anthropic:** `claude-sonnet-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-fable-5`
* **Google:** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
* **Alibaba:** `qwen3.8-max`
* **Zhipu / ZAI:** `glm-5.3`, `glm-5.3-flash`
* **Moonshot AI:** `kimi-k3`

---

## ⚖️ Métrica de Evaluación

* **Balanced Accuracy Score**: Media aritmética del recall obtenido en cada una de las 18 clases.
* **Baseline aleatorio uniforme:** `1 / 18 ≈ 0.0555` (5.55%).
* **Puntuación de referencia en Leaderboard:** `~0.531`.

---

## 🛡️ Estrategia de Validación contra Fugas (Leakage-Free CV)

> **Regla de oro:** En el conjunto de test (`test.csv`), los pasajes pertenecen a **prompts que no aparecen en el entrenamiento (`train.csv`)**.
> Si se utiliza un K-Fold tradicional aleatorio, el modelo memoriza palabras temáticas compartidas entre train y val (fuga masiva de información).

Por tanto, este pipeline utiliza **`StratifiedGroupKFold` de 5 folds agrupando estrictamente por `prompt_id`**, garantizando que ningún prompt visto en validación exista en el entrenamiento de dicho fold.

---

## 🔬 Descubrimientos Estilométricos Clave (Stylometric Audit)

Mediante el script exploratorio `src/eda.py` se identificaron firmas estilísticas deterministas entre las familias:

1. **Raya larga o Em-dash (`—` / `--`):**
   * **Modelos Gemini (Google):** Frecuencia prácticamente nula (`0.00` a `0.01` por texto).
   * **Modelos Claude y GLM:** Usuarios intensivos (`1.26` a `1.62` rayas por pasaje).
   * **Modelos GPT:** Uso moderado (`0.11` a `0.49`).
2. **Dos puntos (`:`):**
   * **Gemini:** Uso muy bajo (`0.17` - `0.23`).
   * **GPT / Kimi:** Uso muy elevado (`0.89` - `0.99`), casi 5 veces superior.
3. **Comillas tipográficas (Curly `“ ” ‘ ’` vs Straight `" '`):**
   * **Claude y GLM:** Utilizan comillas rectas ASCII de forma uniforme (`0.000` de comillas tipográficas).
   * **OpenAI y Kimi:** Tienen una presencia marcada de comillas tipográficas curvadas (`0.70` - `1.27`).
4. **Contracciones (`it's`, `don't`, `can't`):**
   * `claude-sonnet-5` tiene la densidad de contracciones más alta de todo el corpus (`1.63`).
   * `gemini-3.1-pro-preview` mantiene un registro sumamente formal con mínima contracción (`0.19`).

---

## 🏗️ Arquitectura del Pipeline

```text
08_name_that_llm/
├── README.md                          # Documentación del proyecto
├── download_data.py                   # Script de descarga y sincronización de datos
├── submission.parquet                 # Archivo oficial de envío en formato Parquet
├── data/                              # train.csv, test.csv, sample_submission.parquet
├── src/
│   ├── __init__.py
│   ├── eda.py                         # Auditoría estilométrica y cálculo de frecuencias
│   ├── features.py                    # Extracción densa + N-Grams (palabras y caracteres)
│   ├── model.py                       # Validación 5-Fold StratifiedGroupKFold y ensamble
│   └── submit.py                      # Generador y validador de submission.parquet
└── notebooks/
    └── kaggle_inference_notebook.py   # Script autónomo listo para correr en Kaggle GPU/CPU
├── sota_kernel/                       # Pipeline SOTA Kaggle (GPU, multi-LoRA & spectral readouts)
    ├── kernel-metadata.json           # Configuración con 16 datasets y 7 modelos base
    └── name_that_llm_sota_submission.ipynb # Notebook SOTA confirmado (0.642 LB Architecture)
```

### Componentes del Ensamble y Evolución:
* **V1 Baseline (0.304 LB):** Logistic Regression + Calibrated Ridge + LightGBM sobre 32 métricas estilométricas y 85.000 n-gramas.
* **V2 Augmented Multi-Scale Hungarian Ensemble (0.341 LB - Oficial Verificado en Leaderboard):**
  * **Datos aumentados:** Fusión de `train.csv` (2.699) + `passages.csv` (8.549) de `prvsiyan/prvsiyans-extended-name-that-llm-dataset` -> **11.248 pasajes de entrenamiento balanceados**.
  * **Representación multiescala:** 137.448 variables combinando n-gramas de caracteres (2 a 5) y de palabras (1 a 2) con sublinear TF.
  * **Ensamble biproceso calibrado:** RidgeClassifier (alpha=2.5) calibrado con Softmax + SGDClassifier (log_loss, alpha=1e-5).
  * **Clustering temático + Algoritmo Húngaro:** Agrupación jerárquica por distancia coseno (`fcluster`) y emparejamiento bipartito 1-a-1 estricto (`linear_sum_assignment`), garantizando que cada prompt tenga exactamente una predicción por modelo.
* **V3 SOTA All-Public LoRA24 + Soft Sinkhorn + ATLAS Ensemble (0.662 LB - Verificado Oficialmente, Top 3.8% Mundial, Puesto 153 / 3.949):**
  * **Representación profunda multimodelo:** Extracción pooled de estados ocultos en capas intermedias [2, 4, 6, 8, 12, 16, 20, 24] mediante adaptadores LoRA24 de 2 etapas entrenados sobre Qwen3-14B, Qwen3-8B, Llama-3.1-8B, Qwen2.5-14B y DeBERTa-v3-large.
  * **Transporte óptimo Soft Sinkhorn (`reg=0.45`, `target_mix=0.35`):** Sustituye la asignación rígida húngara de 18 clases por una regularización de entropía entrópica suave y adaptativa, evitando penalizar los clusters reducidos (2-5 respuestas) del test set.
  * **Probabilidad condicionada ATLAS 20%:** Re-puntuación por verosimilitud autor-fuente generativa guiada (arXiv 2610.19583) asignada dinámicamente con presupuesto de cómputo en GPU sobre filas dudosas (`margin < 0.30`).
  * **Protección tipográfica y gemelos centrados:** Filtros de spaced em-dashes (`—`) y gaussian twins para neutralizar firmas de formateo artificiales.
* **V4 SOTA Apex: 6-Backbone LoRA + 20% ATLAS + Calibrated Beta-Bernoulli + Roman Tamrazov V40A Sibling-Style Residual (0.664 - 0.670+ LB Target):**
  * **Eliminación de Máscaras Rígidas (`-1e9`):** La auditoría de errores demostró que penalizar rígidamente modelos por marcadores tipográficos (comillas tipográficas, em-dashes) degradaba el score cuando el prompt original contenía la marca y el LLM la repetía. Se eliminan las exclusiones duras en favor de un suavizado Bayesiano calibrado (`1.2`).
  * **Tipografía Bayesiana Jerárquica Beta-Bernoulli Calibrada (37 marcadores):** Suavizado analítico de 37 firmas estilométricas deterministas ponderadas por contraste de señal con filtros de longitud y cobertura con factor de evidencia re-calibrado a `1.2` (evita sobrecorrección en textos cortos).
  * **Residuo Estilístico Intrafamiliar V40A (`sibling_style.py`):** Modelo de regresión estilística OOF mediante 5-fold `GroupKFold` (agrupado por `prompt_id`) sobre las familias disputadas (`claude`, `gemini`, `gpt`, `glm`). Garantiza matemáticamente la conservación del 100% de la masa de probabilidad por proveedor (`logsumexp(out) == logsumexp(old)`), modulando únicamente la identidad de hermanos (`gate = np.clip((0.65 - margin) / 0.65, 0., 1.)`, `max_alpha = 0.08`).
  * **Transporte Óptimo Soft Sinkhorn Calibrado (`reg = 0.60`, `target_mix = 0.35`):** Ajuste fino de la contracción marginal hacia la distribución a priori balanceada del test set.
  * **Reserva de Seguridad (`candidate_reference_0660.parquet`):** Ejecución protegida bajo `try / except` que garantiza fallback automático al candidato base de 0.662+ ante cualquier inconsistencia en tiempo de ejecución.

---

## 🚀 Cómo Ejecutar

1. **Descargar datos:**
   ```bash
   python 08_name_that_llm/download_data.py
   ```

2. **Auditoría estilométrica:**
   ```bash
   python 08_name_that_llm/src/eda.py
   ```

3. **Entrenar y evaluar (5-Fold Group CV):**
   ```bash
   python 08_name_that_llm/src/model.py
   ```

4. **Generar y validar envío oficial:**
   ```bash
   python 08_name_that_llm/src/submit.py
   ```
