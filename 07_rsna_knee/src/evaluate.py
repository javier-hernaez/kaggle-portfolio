"""
Evaluation metrics for RSNA Knee Abnormality Detection.
Computes the competition primary metric: Macro Average ROC-AUC across the 12 targets.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def compute_competition_metric(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    targets: list = TARGET_COLUMNS,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes per-target ROC-AUC and the macro-average ROC-AUC.
    y_true and y_pred must contain the specified target columns.
    """
    per_target_auc = {}
    for target in targets:
        y_t = y_true[target].values
        y_p = y_pred[target].values

        # Ensure no NaNs in true values for evaluation
        valid_mask = ~np.isnan(y_t)
        y_t_clean = y_t[valid_mask]
        y_p_clean = y_p[valid_mask]

        if len(np.unique(y_t_clean)) < 2:
            # Cannot compute AUC if only one class exists
            per_target_auc[target] = 0.5
            continue

        try:
            auc = roc_auc_score(y_t_clean, y_p_clean)
        except Exception:
            auc = 0.5
        per_target_auc[target] = float(auc)

    macro_auc = float(np.mean(list(per_target_auc.values())))
    return macro_auc, per_target_auc


def format_evaluation_summary(macro_auc: float, per_target: Dict[str, float]) -> str:
    lines = [
        "=" * 65,
        "          RSNA KNEE ABNORMALITY EVALUATION SUMMARY              ",
        "=" * 65,
        f"{'Target Abnormality':25s} | {'ROC-AUC':10s}",
        "-" * 65,
    ]
    for target, score in per_target.items():
        lines.append(f"{target:25s} | {score:10.4f}")
    lines.append("-" * 65)
    lines.append(f"{'OVERALL MACRO ROC-AUC':25s} | {macro_auc:10.4f}")
    lines.append("=" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-test with dummy values
    np.random.seed(42)
    n = 58
    dummy_true = pd.DataFrame(np.random.randint(0, 2, size=(n, 12)), columns=TARGET_COLUMNS)
    dummy_pred = pd.DataFrame(np.random.uniform(0.1, 0.9, size=(n, 12)), columns=TARGET_COLUMNS)

    macro, details = compute_competition_metric(dummy_true, dummy_pred)
    print(format_evaluation_summary(macro, details))
