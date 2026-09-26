# 🦿 RSNA Knee Abnormality Detection (2026)

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA_Knee_Abnormality_Detection-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Metric](https://img.shields.io/badge/Metric-Macro_ROC--AUC-success?style=for-the-badge)](https://scikit-learn.org/)
[![Public Score](https://img.shields.io/badge/Public_Score-0.94300-success?style=for-the-badge)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Leaderboard](https://img.shields.io/badge/Rank-%23580_/_4284_(Top_13%25)-blue?style=for-the-badge)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Solución y pipeline reproducible para la competición **RSNA Knee Abnormality Detection AI Challenge (2026)** organizada por la *Radiological Society of North America (RSNA)* en Kaggle.

---

## 📌 1. Descripción del Reto y Objetivos Clínicos

El objetivo es predecir la presencia o ausencia de **12 patologías y anomalías clínicas críticas en la rodilla** a partir de resonancias magnéticas (MRI):

| # | Patología / Anomalía | Descripción Clínica | Plano MRI Principal | Prevalencia (Gold Set) |
| :-: | :--- | :--- | :---: | :---: |
| 01 | **ACL** | Rotura de ligamento cruzado anterior (*Anterior Cruciate Ligament*) | Sagital | 41.4% |
| 02 | **MCL** | Rotura/esguince de ligamento colateral medial (*Medial Collateral Ligament*) | Coronal | 15.5% |
| 03 | **Medial Meniscus** | Rotura de menisco interno / medial | Sagital / Coronal | 44.8% |
| 04 | **Lateral Meniscus** | Rotura de menisco externo / lateral | Sagital / Coronal | 39.7% |
| 05 | **Medial OA** | Artrosis de compartimento femorotibial medial | Coronal | 25.9% |
| 06 | **Lateral OA** | Artrosis de compartimento femorotibial lateral | Coronal | 19.0% |
| 07 | **PF OA** | Artrosis femoropatelar (*Patellofemoral Osteoarthritis*) | Axial / Sagital | 36.2% |
| 08 | **Effusion** | Derrame articular / hidrartrosis | Sagital / Axial | 60.3% |
| 09 | **Synovitis** | Sinovitis / engrosamiento y proliferación sinovial | Axial / Sagital | 46.6% |
| 10 | **Baker's** | Quiste de Baker / quiste poplíteo | Sagital / Axial | 20.7% |
| 11 | **Contusion** | Contusión ósea / edema óseo trabecular (bone bruise) | Coronal / Sagital | 32.8% |
| 12 | **Fracture** | Fractura aguda / oculta / avulsión cortical | Coronal / Sagital | 31.0% |

### Métrica de Evaluación
La métrica oficial es el **Macro Average Area Under the ROC Curve (Macro ROC-AUC)** sobre las 12 anomalías:

$$\text{Macro ROC-AUC} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$

---

## 🔬 2. Particularidad Fundamental: Supervisión Débil (*Weak Supervision*)

El conjunto de datos presenta una arquitectura semi-supervisada:
* **Total de estudios en Train:** 4.407 estudios (24.371 series de MRI).
* **Estudios con etiqueta humana oro (Gold-Standard):** Solo **58 estudios (1.32%)**.
* **Estudios sin etiquetas binarias pero con informe clínico:** **4.349 estudios (98.68%)**.
* **Informes clínicos:** Texto libre multilingüe (español, inglés, francés, alemán, portugués) redactado por radiólogos de 19 centros hospitalarios mundiales.
* **Test set:** En inferencia **NO hay informes clínicos**, únicamente las secuencias de MRI en formato DICOM (`test_series/`).

---

## 💡 3. Hallazgos del Análisis Exploratorio (EDA)

1. **Co-ocurrencias Clínicas Fuertes:**
   * $\text{Medial OA} \leftrightarrow \text{Baker's}$ ($r = +0.48$): Asociación degenerativa clásica entre artrosis y quiste de Baker.
   * $\text{Medial Meniscus} \leftrightarrow \text{Medial OA}$ ($r = +0.42$): Las roturas meniscales degenerativas acompañan la gonartrosis medial.
   * $\text{ACL} \leftrightarrow \text{Contusion}$ ($r = +0.38$): Patrón clásico de *bone bruise* en cóndilo femoral lateral y meseta tibial posterior tras mecanismo de pivote.
   * $\text{Effusion} \leftrightarrow \text{Synovitis}$ ($r = +0.40$): Correlación fisiopatológica de sinovitis activa con aumento de líquido articular.
2. **Distribución de Planos de Adquisición MRI:**
   * **Sagital:** 40.5%
   * **Coronal:** 35.3%
   * **Axial:** 24.2%
   * Promedio de 5.53 series por estudio (entre 3 y 14 series).

---

## 🛠️ 4. Arquitectura del Pipeline Desarrollado

```text
07_rsna_knee/
├── data/
│   ├── sample_submission.csv        # Template de envío oficial (12 targets)
│   ├── test.csv                     # UIDs de estudios de test
│   ├── test_series.csv              # Metadatos de series de test
│   ├── train.csv                    # 4.407 estudios (58 gold + 4.349 informes)
│   └── train_series.csv             # 24.371 metadatos de series de entrenamiento
├── eda_outputs/
│   ├── target_correlations.csv      # Matriz de correlación entre patologías
│   └── target_prevalence_58_gold.csv# Prevalencia empírica en gold standard
├── src/
│   ├── __init__.py
│   ├── eda.py                       # Análisis exploratorio y estadístico
│   ├── report_extractor.py          # Extractor multilingüe de reglas/NLP clínico
│   ├── series_features.py           # Feature engineering de protocolos MRI
│   ├── evaluate.py                  # Cálculo exacto de Macro ROC-AUC
│   ├── model.py                     # Ensamble de predicción multi-etiqueta
│   └── submission.py                # Validador y generador de envíos
├── download_data.py                 # Verificador de archivos Kaggle
├── baseline_submission.csv          # Archivo de predicción verificado
└── README.md
```

### Componentes Implementados:

1. **`report_extractor.py` (Weak Supervision NLP):**
   * Diccionario léxico multilingüe especializado en resonancia de rodilla (inglés, español, francés, alemán, portugués).
   * Detección de patrones de negación ("no sign of", "sin desgarro", "keine Ruptur", "conservado", "intacto").
   * **Rendimiento evaluado en los 58 casos Gold-Standard:**
     * **MCL:** `0.8730` ROC-AUC
     * **Lateral Meniscus:** `0.8248` ROC-AUC
     * **ACL:** `0.7941` ROC-AUC
     * **Baker's:** `0.7862` ROC-AUC
     * **Medial Meniscus:** `0.7422` ROC-AUC
     * **Fracture:** `0.7375` ROC-AUC
     * **Macro ROC-AUC Global:** **`0.6999`**
2. **`series_features.py` (Tabular MRI Features):**
   * Conteo de secuencias por plano (axial, coronal, sagital).
   * Ratios de supresión grasa (*Fat Suppression*) y sensibilidad a fluidos (*Fluid Sensitive*).
   * Protocolos específicos (Sagittal T2-FS, Coronal PD-FS, Axial T2-FS).
3. **`model.py` (Multi-Label Classifier & Prior Calibration):**
   * Modela los 4.407 estudios aprovechando las pseudo-etiquetas de supervisión débil.
   * Calibra las probabilidades predichas con las tasas base empíricas del conjunto oro.
4. **`submission.py` (Generador de Envíos):**
   * Comprueba concordancia exacta de IDs, 13 columnas requeridas, probabilidades continuas en $[0, 1]$ y ausencia de nulos.
   * Genera el archivo final `baseline_submission.csv`.

---

## 🚀 5. Ejecución Rápida y Reproducción

1. **Verificar o descargar datos:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\download_data.py
   ```

2. **Ejecutar análisis exploratorio (EDA):**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\src\eda.py
   ```

3. **Evaluar el extractor de supervisión débil en el Gold Set:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\src\report_extractor.py
   ```

4. **Entrenar modelos y generar submission:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\src\submission.py
   ```

---

## 🏆 5. Historial de Envíos y Progresión en Leaderboard

| Versión | Arquitectura / Enfoque | Macro ROC-AUC | Estado | Posición Leaderboard |
| :---: | :--- | :---: | :---: | :---: |
| **v2** | Weak Supervision Clinical NLP + Series Priors | `0.500` | Baseline inicial | Exploración |
| **v3** | Multimodal DICOM Vision Central Slices + Protocol Features | `0.426` | Descartado (inversión de correlaciones) | - |
| **v4** | DINOv2-small ViT + Multi-Slot Attention (20 checkpoints) | `0.891` | Salto de rendimiento masivo | Top 60% (#2688) |
| **v5** | Multi-Model Ensemble: DINOv2 + RadImageNet + Raptor + CoAtNet | `0.942` | Top tier competitivo | #652 / 4.284 (Top 15%) |
| **v6** | DINOsaur V5 Grandmaster (DINOv2 + Rad + Raptor + CoAtNet + ConvNeXt + Calibrador OOF) | `0.941` | Análisis: calibrador lineal OOF generaba ligero drift | Top 15% |
| **v7** | DINOsaur V5 Final: Speedy Raptors v34 (DINOv2 20ckpts + RadImageNet + 4xCoAtNet) + Depth-TTA | **`0.943`** | Confirmado en Leaderboard | **#580 / 4.284 (Top 13%)** |
| **v8** | **DINOsaur V5 SOTA:** Rank-Logit Fusion + Repair-v1 CoAt Residual + Synovitis Rad Rescue + Comorbidity Lift | *En ejecución GPU* | Estado del arte público | **Objetivo: Top 5-10% (0.944 - 0.946)** |

---

## 🚀 6. Ejecución Rápida y Reproducción

1. **Verificar o descargar datos:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\download_data.py
   ```

2. **Ejecutar análisis exploratorio (EDA):**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\src\eda.py
   ```

3. **Evaluar el extractor de supervisión débil en el Gold Set:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\src\report_extractor.py
   ```

4. **Monitorizar ejecución y auto-envío en Kaggle GPU:**
   ```powershell
   .\.venv\Scripts\python.exe 07_rsna_knee\submit_when_ready.py
   ```

---

## 🔮 7. Arquitectura del Gran Ensamble Multimodal (v8)

* **Backbones de Visión Médica y Auto-Supervisión:**
  * `DINOv2-small` (`metaresearch/dinov2`): 20 modelos con Slot Attention anatómico para cortes Sagitales, Coronales y Axiales (prefix freezing optimizado).
  * `RadImageNet ResNet-50`: Red preentrenada en millones de imágenes médicas con 10 cabezales de atención (`E10`, `E11`, `E13`).
  * `Raptor CoAtNet` (4 vistas): Modelos RMLP-2 en resoluciones nativas 336px y 384px con enrutamiento de ventanas de alta capacidad.
  * `CoAtNet Readers` complementarios:
    - Residual Gated CoAtNet (epochs 4, 6, 8).
    - D4 Depth-Zone SWA3 (adaptación a tres zonas de profundidad por corte).
    - Global96 full-width gated reader.
    - **Repair-v1 CoAtNet:** Inyectado como residuo de diversidad del 15% (`_raptor_repair15.csv`).
* **Innovaciones Algorítmicas Clave (v8):**
  1. **Fusión Rank-Logit:** Mapeo de percentiles a espacio logit $z = \log(r / (1 - r))$, combinación ponderada y sigmoide inverso, evitando el aplanamiento de colas de confianza propio del promedio lineal de rangos.
  2. **Reparación Geométrica A5:** Ordenación física de cortes mediante proyección del vector normal de corte ($IPP \cdot (IOP_{row} \times IOP_{col})$) activada únicamente cuando el `InstanceNumber` DICOM presenta anomalías de monotonicidad.
  3. **Rescate RadImageNet para Sinovitis:** Calibración específica reduciendo la cuota Raptor/CoAt a 0.55 en Sinovitis, donde RadImageNet exhibe mayor sensibilidad a hipertrofia sinovial.
  4. **Lift Clínico de Comorbilidad:** Inyección de correlaciones cruzadas observadas en resonancia ortopédica (Contusión $\rightarrow$ Fractura, Artrosis Medial $\rightarrow$ Lateral, Artrosis Lateral $\rightarrow$ Menisco Externo, LCA $\rightarrow$ LCM, Derrame $\rightarrow$ Sinovitis).

