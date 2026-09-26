# 🏆 Kaggle Machine Learning Portfolio

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-green?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Yandex-yellow?style=for-the-badge)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-DMLC-red?style=for-the-badge)](https://xgboost.readthedocs.io/)

Repositório con soluciones, ingeniería de variables y modelos de Machine Learning desarrollados para competiciones de **Kaggle**. Cada carpeta contiene código modular, reproducible y documentado.

---

## 📈 Competiciones y Puntuaciones

| # | Competición | Tipo | Mejor Modelo | Métrica | Score Público (Kaggle) | Posición / Percentil | Estado |
| :-: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 01 | [**Titanic: Machine Learning from Disaster**](01_titanic/) | Clasificación Binaria | Woman-Child-Group (WCG) | Accuracy | **`0.81578`** | **#246 / 9.639 (Top 2.5%)** | ✅ Completada |
| 02 | [**Spaceship Titanic**](02_spaceship_titanic/) | Clasificación Binaria | Ensemble (LightGBM + CatBoost) | Accuracy | **`0.80500`** | **#473 / 1.536** | ✅ Completada |
| 03 | [**House Prices: Advanced Regression**](03_house_prices/) | Regresión Continua | Blend (Ridge + Lasso + CB + LGB) | RMSLE | **`0.12589`** | **#943 / 3.178** | ✅ Completada |
| 04 | [**Predicting EV Purchases (s6e9)**](04_predicting_ev_purchases/) | Clasificación Binaria | Multi-Anchor Consensus v8 (Top-3 Anchors + 10F XGB) | ROC-AUC | **`0.94651`** | **#45 / 2.217 (Top 2.0%)** | 🚀 En competición |
| 05 | [**NLP with Disaster Tweets**](05_disaster_tweets/) | Clasificación de Texto (NLP) | Ensamble (LR + Ridge + Threshold Opt) | F1-Score | **`0.81336`** | **#214 / 469** | 🚀 En competición |
| 06 | [**Kaggriculture**](06_kaggriculture/) | Simulación y Estrategia Económica | Industrial Strategic Agent v3 (Wheat Reserve Fix + Sheep Diversification + Anti-Crash) | Winrate / Coins | **100% WR vs v2 (\$56k - \$66k media)** | *Leaderboard Activo* | 🚀 En competición |
| 07 | [**RSNA Knee Abnormality Detection**](07_rsna_knee/) | Visión Médica & Transformers | DINOsaur V5 v8 (Rank-Logit Fusion + Repair-v1 CoAt Residual + Synovitis Rad Rescue + Comorbidity Lift) | Macro ROC-AUC | **`0.94300`** *(v8 en GPU)* | **#580 / 4.284 (Top 13%)** | 🚀 En competición |

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python 3.12
* **Manipulación y análisis de datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **Algoritmos de Machine Learning:**
  * Árboles y ensambles: `RandomForestClassifier`, `VotingClassifier` (`scikit-learn`)
  * Gradient Boosting: `LightGBM` (Microsoft), `CatBoost` (Yandex), `XGBoost` (DMLC)
* **Validación y Métricas:** Validación cruzada estratificada de 5 folds (*Stratified K-Fold CV*)
* **Gestión de envíos:** Kaggle API CLI

---

## ⚙️ Configuración del Entorno Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/kaggle-portfolio.git
   cd kaggle-portfolio
   ```

2. **Crear y activar el entorno virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar credenciales de Kaggle:**
   Coloca tu archivo `kaggle.json` en `~/.kaggle/kaggle.json` (o guarda tu API token en `~/.kaggle/access_token`).

---

## 📜 Estructura del Repositorio

```text
kaggle-portfolio/
├── .gitignore                      # Exclusiones de Git (entornos virtuales, caches)
├── README.md                       # Índice y presentación general del portafolio
├── requirements.txt                # Dependencias unificadas del proyecto
│
├── 01_titanic/                     # Proyecto Titanic (Score: 0.81578 - Top 2.5%)
│   ├── README.md                   # Documentación técnica del modelo WCG
│   ├── download_data.py            # Descarga de datos
│   └── src/                        # Código fuente de los modelos
│
├── 02_spaceship_titanic/           # Proyecto Spaceship Titanic (Score: 0.80500)
│   ├── README.md
│   └── src/model.py                # Ensamble LightGBM + CatBoost
│
├── 03_house_prices/                # Proyecto House Prices (Score: 0.12589)
│   ├── README.md
│   └── src/model.py                # Stacking / Blending Regresores
│
└── 04_predicting_ev_purchases/     # Proyecto Playground s6e9: EV Purchases
    ├── README.md                   # Documentación y análisis de función generadora
    ├── download_data.py            # Descarga de datos
    └── src/
        ├── eda.py                  # Análisis exploratorio y correlaciones
        ├── features.py             # Feature engineering & DGP Probit
        └── model.py                # Ensamble 5-Fold LGBM + CatBoost + XGBoost
│
└── 05_disaster_tweets/             # Proyecto NLP: Clasificación de Tweets de Desastre
    ├── README.md                   # Documentación y análisis estilométrico
    ├── download_data.py            # Descarga de datos
    └── src/
        ├── eda.py                  # Análisis exploratorio y keywords
        ├── text_cleaner.py         # Limpieza, entidades HTML y mojibake
        ├── features.py             # TF-IDF (palabras + caracteres) y meta-features
        ├── evaluate.py             # Optimización de umbral de F1-Score
        ├── baseline_model.py       # 5-Fold CV (LR + Calibrated Ridge)
        └── train_transformer_kaggle.py # Pipeline GPU DeBERTa-v3
│
└── 06_kaggriculture/               # Simulación y Estrategia Económica de Granjas
    ├── README.md                   # Documentación del juego, economía y heurísticas
    ├── main.py                     # Agente autónomo para envío a Kaggle
    ├── evaluate.py                 # Benchmark de simulación 720 turnos vs baselines
    ├── submit.py                   # Script de validación y subida a Kaggle API
    ├── simulator/                  # Motor de simulación puro (zero dependencies)
    │   ├── engine.py               # Lógica de juego, mercado y turnos
    │   └── battle.py               # Torneos y métricas de victoria
    └── src/
        ├── heuristic_agent.py      # Agente estratégico (rotación, cuadrantes, jornaleros)
        └── starter_baseline.py     # Agente baseline oficial de Kaggle
│
└── 07_rsna_knee/                   # RSNA Knee Abnormality Detection (Multimodal & Weak Supervision)
    ├── README.md                   # Documentación médica, análisis clínico y roadmap
    ├── download_data.py            # Descarga y verificación de metadatos
    ├── baseline_submission.csv     # Envío inicial verificado y calibrado
    └── src/
        ├── eda.py                  # Exploración de prevalencias y correlaciones clínicas
        ├── report_extractor.py     # Extractor multilingüe NLP de supervisión débil (0.6999 ROC-AUC)
        ├── series_features.py      # Feature engineering de protocolos MRI (planos, contrastes)
        ├── evaluate.py             # Métrica oficial Macro ROC-AUC
        ├── model.py                # Clasificador multi-etiqueta con calibración
        └── submission.py           # Validador y generador de envíos oficiales
```

---

## 👤 Autor

* **Javier Hernáez** — [GitHub (@javier-hernaez)](https://github.com/javier-hernaez) — [Kaggle](https://www.kaggle.com)
