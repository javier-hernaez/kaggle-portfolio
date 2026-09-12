"""
Módulo de evaluación y optimización de umbral de decisión para F1-Score.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

def find_best_threshold(y_true, y_probs, step=0.005):
    """
    Realiza una búsqueda lineal exhaustiva para encontrar el umbral de corte
    óptimo que maximiza el F1-Score binario en el conjunto de validación OOF.
    """
    thresholds = np.arange(0.15, 0.85, step)
    best_thresh = 0.50
    best_f1 = 0.0

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    # Métricas en umbral por defecto (0.50)
    preds_def = (y_probs >= 0.50).astype(int)
    f1_def = f1_score(y_true, preds_def, zero_division=0)
    acc_def = accuracy_score(y_true, preds_def)

    # Métricas en umbral óptimo
    preds_opt = (y_probs >= best_thresh).astype(int)
    acc_opt = accuracy_score(y_true, preds_opt)
    prec_opt = precision_score(y_true, preds_opt, zero_division=0)
    rec_opt = recall_score(y_true, preds_opt, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    print("=" * 60)
    print(" EVALUACION Y OPTIMIZACION DE UMBRAL (F1-SCORE)")
    print("=" * 60)
    print(f"ROC-AUC Score          : {auc:.5f}")
    print(f"Umbral por defecto 0.50 : F1 = {f1_def:.5f} | Acc = {acc_def:.5f}")
    print(f"Umbral óptimo ({best_thresh:.3f}) : F1 = {best_f1:.5f} | Acc = {acc_opt:.5f}")
    print(f" - Precisión            : {prec_opt:.5f}")
    print(f" - Recall               : {rec_opt:.5f}")
    print(f" - Ganancia de F1       : {best_f1 - f1_def:+.5f}")
    print("=" * 60)

    return best_thresh, best_f1, auc
