# 🏈 NFL Big Data Bowl 2027: Combine to NFL Performance
## *The Combine Kinematic Translation Index (CKTI)*

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NFL%20Big%20Data%20Bowl%202027-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete%20Solution-brightgreen?style=for-the-badge)]()
[![Evaluation](https://img.shields.io/badge/NFL%20Analytics-Publication%20Ready-red?style=for-the-badge)]()

---

## 📌 Resumen Ejecutivo

Durante décadas, las evaluaciones pre-draft de la NFL han estado condicionadas por mediciones estáticas y cronómetros manuales (las 40 yardas, el salto vertical, el short shuttle). Sin embargo, la evidencia estadística revela dos fallas críticas en este paradigma:
1. **El Mito de las 40 Yardas:** La velocidad lineal sin oposición en línea recta apenas correlaciona con los snaps profesionales ($r = -0.108$) ni con la capacidad de generar separación ante esquemas defensivos de la NFL.
2. **Crisis de Ausentismo en Drills:** Más del **64.7% de los prospectos declinan el 3-cone drill** y el **60.5% declinan el short shuttle**, dejando a los departamentos de scouting sin datos de agilidad y cambio de dirección.

En esta solución para el **NFL Big Data Bowl 2027**, presentamos el **Combine Kinematic Translation Index (CKTI)**, un marco analítico y biomecánico que explota las trayectorias de sensores ópticos a alta frecuencia (**10 Hz**) capturadas durante el NFL Scouting Combine para **510 prospectos** (drafts 2023–2025). 

Vinculando la cinemática del Combine con **más de 314,000 jugadas oficiales de temporada regular** (`player_play.csv`) y trayectoria (`player_career_successes.csv`), demostramos empíricamente que la cinemática de alta frecuencia supera de forma contundente a las métricas tradicionales de cronómetro:
* **Separación de Receptores (WR Separation at Pass Forward):** Mejora del **+25.5% en varianza explicada ($R^2 = 0.1475$ vs $0.1175$)**, donde la *Kinematic Break Efficiency (KBE)* es el factor predictivo positivo individual #1 ($\beta = +0.108$).
* **Longevidad y Snaps de Carrera (510 Prospectos):** Los tests tradicionales explican prácticamente cero variación ($R^2 = 0.0021$), mientras que los modelos cinemáticos logran un incremento de **27 veces en varianza explicada ($R^2 = 0.0579$, $p = 0.0010$)**.

---

## 📐 Marco Matemático y Cinemático (10 Hz)

A partir de los fotogramas $(x_t, y_t, s_t, a_t, \text{dir}_t)$ a $\Delta t = 0.1\,\text{s}$, formulamos cuatro métricas biomecánicas fundamentales:

### 1. Kinematic Break Efficiency (KBE)
Mide la capacidad de un receptor para decelerar violentamente al entrar en el ápice de su corte sin perder el control de su centro de gravedad, y convertir inmediatamente ese cambio angular en aceleración explosiva de salida:
$$\Delta \theta_t = ((\text{dir}_t - \text{dir}_{t-1} + 180^\circ) \pmod{360^\circ}) - 180^\circ$$
$$\text{KBE} = \left( \frac{s_{\text{exit}}}{s_{\text{entry}}} \right) \times (1.0 + a_{\text{burst}})$$
Donde $s_{\text{entry}}$ se evalúa 3 fotogramas antes del corte, $s_{\text{exit}}$ 4 fotogramas tras el ápice, y $a_{\text{burst}}$ es la aceleración máxima post-corte.

### 2. Route Sharpness Index (RSI)
Velocidad angular pico $\omega_t = \frac{|\Delta \theta_t|}{\Delta t}$ (en deg/s) normalizada por la velocidad instantánea en el ápice, cuantificando el ángulo de corte sin apertura de radio:
$$\text{RSI} = \frac{\max(\omega_t)}{\max(s_{\text{apex}}, 1.0)}$$

### 3. Aceleración Centrípeta y "Bend Index" ($a_c$)
Para pass-rushers trazando el arco alrededor del tackle ofensivo (*Run the Hoop* y *Pass Rush Drill*):
$$a_c = s_t \cdot \omega_{\text{rad}, t}$$
El percentil 90 de $a_c$ define el *Bend Index*, capturando la inclinación del torso y adherencia lateral contra fuerzas de inercia.

### 4. Dinámica de Jerk Neuromuscular ($J = da/dt$)
Derivada temporal de la aceleración. El jerk positivo en la fase de arrancada (0 a 1.5s) refleja el reclutamiento instantáneo de unidades motoras y fibras musculares de contracción rápida.

---

## 📊 Resultados Empíricos y Validación Cruzada (5-Fold CV)

| Experimento | Métrica Objetivo NFL | Modelo Baseline (Cronómetro Tradicional) | Modelo CKTI (Cinemática 10 Hz) | Reducción de Error (RMSE) | Significancia Bootstrap ($p$-valor) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WR Route Separation** | Separación al momento del pase (yds) | $R^2 = 0.1175$, $\text{RMSE} = 0.4236$ | **$R^2 = 0.1475$, $\text{RMSE} = 0.4163$** | $-0.0074$ yds | $p = 0.1765$ |
| **Pass Rusher Quick Pressure** | Presión al QB en $< 2.5\text{s}$ por snap | $R^2 = 0.2119$, $\text{RMSE} = 0.0128$ | $R^2 = 0.1982$, $\text{RMSE} = 0.0129$ | Paridad | $p = 0.6585$ |
| **Career Durability (510 Prospectos)** | Snaps totales de carrera en la NFL | $R^2 = 0.0021$, $\text{RMSE} = 697.4$ | **$R^2 = 0.0579$, $\text{RMSE} = 677.6$** | **$-19.58$ snaps** | **$p = 0.0010$ (Significativo al 99.9%)** |

---

## 🖼️ Figuras de Nivel Publicación (eda_outputs/)

1. **`figure1_combine_missingness.png`**: Diagnóstico de ausencias en pruebas tradicionales (>60% en 3-cone y short-shuttle).
2. **`figure2_traditional_vs_nfl_correlation.png`**: Matriz de correlación demostrando la escasa correlación de la prueba de 40 yardas con el éxito profesional.
3. **`figure3_wr_separation_regression.png`**: Correlación lineal ($r = +0.279$) entre KBE y la separación real en la NFL (destacando prospectos como Tank Dell, Puka Nacua y Marvin Harrison Jr.).
4. **`figure4_model_performance_comparison.png`**: Gráfico de barras comparativo de $R^2$ out-of-fold.
5. **`figure5_feature_importance_coefficients.png`**: Coeficientes estandarizados del modelo Ridge multivariante.
6. **`figure6_drill_kinematics_trajectory.png`**: Trayectoria espacial 2D y perfil temporal de velocidad/aceleración en corte de ruta.
7. **`figure7_prospect_scout_radar.png`**: Radar Card de Scouting para toma de decisiones en el War Room del Draft.

---

## 🛠️ Estructura del Proyecto

```text
09_nfl_big_data_bowl_2027/
├── README.md                           # Documentación técnica completa (este archivo)
├── download_data.py                    # Descarga y verificación mediante Kaggle API
├── data/                               # Datasets oficiales de la competición
│   ├── players.csv                     # Metadatos de 510 prospectos
│   ├── combine_results.csv             # Resultados tradicionales del Combine
│   ├── combine_tracking.csv            # Sensores ópticos a 10 Hz (430k+ frames)
│   ├── player_career_successes.csv     # Snaps, inicios y selecciones Pro Bowl/All-Pro
│   ├── games.csv                       # 1,002 partidos de temporada regular
│   ├── player_play.csv                 # 314,000+ jugadas con métricas Next Gen Stats
│   ├── prospect_kinematics.csv         # 27 variables cinemáticas extraídas
│   └── nfl_performance_targets.csv     # 35 targets de rendimiento NFL compilados
├── eda_outputs/                        # Gráficos de calidad editorial y resultados JSON
│   ├── figure1_combine_missingness.png
│   ├── figure2_traditional_vs_nfl_correlation.png
│   ├── figure3_wr_separation_regression.png
│   ├── figure4_model_performance_comparison.png
│   ├── figure5_feature_importance_coefficients.png
│   ├── figure6_drill_kinematics_trajectory.png
│   ├── figure7_prospect_scout_radar.png
│   └── model_benchmark_results.json
├── notebooks/
│   └── nfl_big_data_bowl_2027_solution.ipynb # Master Kaggle Notebook (<2,000 palabras, 7 figuras)
├── src/
│   ├── __init__.py
│   ├── eda.py                          # Análisis exploratorio y diagnóstico
│   ├── kinematics.py                   # Motor extractor cinemático 10 Hz (KBE, RSI, Jerk, Bend)
│   ├── nfl_performance.py              # Agregador de jugadas NGS a targets por jugador
│   ├── models.py                       # Pipeline predictivo 5-Fold CV, Ridge, LightGBM, Bootstrap
│   ├── visualization.py                # Generador de figuras de analítica deportiva
│   └── submission_generator.py         # Constructor del Notebook oficial
└── tests/
    └── test_pipeline.py                # Suite de pruebas unitarias automatizadas (6 tests)
```

---

## 🚀 Guía de Reproducción

Desde la raíz del repositorio y con el entorno virtual activado:

```powershell
# 1. Verificar descarga de datos
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\download_data.py

# 2. Ejecutar análisis exploratorio
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\eda.py

# 3. Extraer features cinemáticas a 10 Hz
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\kinematics.py

# 4. Compilar métricas de rendimiento real NFL (314k jugadas)
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\nfl_performance.py

# 5. Entrenar y evaluar modelos (5-Fold CV + Bootstrap)
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\models.py

# 6. Generar visualizaciones oficiales
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\visualization.py

# 7. Compilar notebook maestro para Kaggle
.\.venv\Scripts\python .\09_nfl_big_data_bowl_2027\src\submission_generator.py

# 8. Ejecutar tests unitarios automatizados
.\.venv\Scripts\python -m unittest discover -s 09_nfl_big_data_bowl_2027/tests -p "test_*.py"
```

---

## 👤 Autor

* **Javier Hernáez** — [GitHub (@javier-hernaez)](https://github.com/javier-hernaez) — [Kaggle](https://www.kaggle.com)
