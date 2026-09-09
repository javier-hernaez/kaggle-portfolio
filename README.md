# 🏆 Kaggle Machine Learning Portfolio

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-green?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Yandex-yellow?style=for-the-badge)](https://catboost.ai/)

Repositorio con soluciones, ingeniería de variables y modelos de Machine Learning desarrollados para competiciones de **Kaggle**. Cada carpeta contiene código modular, reproducible y documentado.

---

## 📈 Competiciones y Puntuaciones

| # | Competición | Tipo | Mejor Modelo | Métrica | Score Público (Kaggle) | Posición / Percentil | Estado |
| :-: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 01 | [**Titanic: Machine Learning from Disaster**](01_titanic/) | Clasificación Binaria | Woman-Child-Group (WCG) | Accuracy | **`0.81578`** | **#246 / 9.639 (Top 2.5%)** | ✅ Completada |
| 02 | [**Spaceship Titanic**](02_spaceship_titanic/) | Clasificación Binaria | Ensemble (LightGBM + CatBoost) | Accuracy | **`0.80500`** | **#473 / 1.536** | ✅ Completada |
| 03 | [**House Prices: Advanced Regression**](03_house_prices/) | Regresión Continua | Blend (Ridge + Lasso + CB + LGB) | RMSLE | **`0.12589`** | **#943 / 3.178** | ✅ Completada |

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Lenguaje:** Python 3.12
* **Manipulación y análisis de datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **Algoritmos de Machine Learning:**
  * Árboles y ensambles: `RandomForestClassifier`, `VotingClassifier` (`scikit-learn`)
  * Gradient Boosting: `LightGBM` (Microsoft), `CatBoost` (Yandex)
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
│   ├── data/                       # Archivos train/test
│   ├── notebooks/                  # Análisis exploratorio (EDA)
│   ├── src/                        # Código fuente de los modelos
│   │   ├── baseline_model.py
│   │   ├── advanced_model.py
│   │   └── wcg_model.py            # Modelo campeón
│   └── submissions/                # Historial de archivos de envío
│
└── 02_spaceship_titanic/           # Proyecto Spaceship Titanic
    └── README.md
```

---

## 👤 Autor

* **Javier Hernáez** — [GitHub (@javier-hernaez)](https://github.com/javier-hernaez) — [Kaggle](https://www.kaggle.com)
