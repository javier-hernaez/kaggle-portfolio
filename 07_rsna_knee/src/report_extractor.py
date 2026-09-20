"""
Multilingual Clinical NLP Report Extractor for RSNA Knee Abnormality Detection.
Extracts weak supervision labels [0.0 - 1.0] from free-text radiology reports
across English, Spanish, French, German, and Portuguese.
Evaluates precision, recall, and ROC-AUC on the 58 gold-standard cases.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eda_outputs"

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

# Multilingual target entity keywords
TARGET_PATTERNS = {
    "ACL": [
        r"\bacl\b",
        r"\blca\b",
        r"\bvkb\b",
        r"anterior\s+cruciate",
        r"cruzado\s+anterior",
        r"crois[eé]\s+ant[eé]rieur",
        r"vorderes?\s+kreuzband",
    ],
    "MCL": [
        r"\bmcl\b",
        r"\blcm\b",
        r"\blci\b",
        r"medial\s+collateral",
        r"colateral\s+medial",
        r"colateral\s+interno",
        r"collat[eé]ral\s+m[eé]dial",
        r"collat[eé]ral\s+interne",
        r"mediales?\s+kollateralband",
        r"innenband",
    ],
    "Medial Meniscus": [
        r"medial\s+menisc",
        r"menisco\s+medial",
        r"menisco\s+interno",
        r"m[eé]nisque\s+m[eé]dial",
        r"m[eé]nisque\s+interne",
        r"innenmeniskus",
        r"medialer?\s+meniskus",
        r"\bmm\b",
    ],
    "Lateral Meniscus": [
        r"lateral\s+menisc",
        r"menisco\s+lateral",
        r"menisco\s+externo",
        r"m[eé]nisque\s+lat[eé]ral",
        r"m[eé]nisque\s+externe",
        r"au[ss]enmeniskus",
        r"lateraler?\s+meniskus",
        r"\blm\b",
    ],
    "Medial OA": [
        r"medial.*(?:osteoarthr|artrosis|gonartr|gonarthr|arthrose|narrowing|pinzamiento|chondromalac|condropat)",
        r"(?:osteoarthr|artrosis|gonartr|gonarthr|arthrose).*medial",
        r"f[eé]morotibial\s+medial",
        r"compartimento\s+medial.*(?:desgaste|artrosis|condr)",
    ],
    "Lateral OA": [
        r"lateral.*(?:osteoarthr|artrosis|gonartr|gonarthr|arthrose|narrowing|pinzamiento|chondromalac|condropat)",
        r"(?:osteoarthr|artrosis|gonartr|gonarthr|arthrose).*lateral",
        r"f[eé]morotibial\s+lateral",
        r"compartimento\s+lateral.*(?:desgaste|artrosis|condr)",
    ],
    "PF OA": [
        r"patellofemoral",
        r"femoropatel",
        r"f[eé]moro-patell",
        r"retropatel",
        r"trochlea",
        r"rotulian",
        r"artrosis.*patel",
        r"patellar\s+arthrosis",
    ],
    "Effusion": [
        r"effusion",
        r"derrame",
        r"[eé]panchement",
        r"gelenkerguss",
        r"erguss",
        r"hidrartrosis",
        r"joint\s+fluid",
        r"liquide\s+intra-articulaire",
    ],
    "Synovitis": [
        r"synovit",
        r"sinovit",
        r"synovial\s+thick",
        r"synoviale",
        r"engrosamiento\s+sinovial",
        r"prolif[eé]ration\s+synoviale",
        r"synovialitis",
    ],
    "Baker's": [
        r"baker",
        r"popliteal\s+cyst",
        r"kyste\s+poplit[eé]",
        r"quiste\s+popl[ií]teo",
        r"quiste\s+de\s+baker",
        r"kyste\s+de\s+baker",
        r"baker-zyste",
        r"bakerzyste",
        r"popliteazyste",
    ],
    "Contusion": [
        r"contusion",
        r"contusi[oó]n",
        r"bone\s+bruise",
        r"edema\s+[oó]seo",
        r"oedema\s+osseux",
        r"[oœ]d[eè]me\s+osseux",
        r"bone\s+marrow\s+edema",
        r"knochen[oö]dem",
        r"knochenkontusion",
        r"marrow\s+contusion",
        r"edema\s+trabecular",
    ],
    "Fracture": [
        r"fractur",
        r"fractura",
        r"fraktur",
        r"break",
        r"arrachement\s+osseux",
        r"avulsi[oó]n",
        r"avulsion",
        r"cortical\s+disruption",
        r"segond",
    ],
}

# Negation indicators across languages
NEGATION_WORDS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnon\b",
    r"\bsin\b",
    r"\bsans\b",
    r"\bkein\b",
    r"\bkeine\b",
    r"\bkeinen\b",
    r"\bohne\b",
    r"\bsem\b",
    r"\bintact\b",
    r"\bintacto\b",
    r"\b[ií]ntegro\b",
    r"\bconservado\b",
    r"\bconserv[eé]\b",
    r"\bpreservado\b",
    r"\bpreserved\b",
    r"\bnormal\b",
    r"\bunremarkable\b",
    r"\bregelrecht\b",
    r"\bunauff[aä]llig\b",
    r"\bnegative\b",
    r"\bnegativo\b",
    r"\bfree\s+of\b",
    r"\blibre\s+de\b",
    r"\bpas\s+de\b",
    r"\babsence\s+de\b",
    r"\baus[eê]ncia\s+de\b",
    r"\bdescartar?\b",
    r"\brules?\s+out\b",
    r"\bno\s+evidence\b",
    r"\bsin\s+evidencia\b",
    r"\bsin\s+signos\b",
]

# Positive confirmation indicators across languages
POSITIVE_WORDS = [
    r"\btear\b",
    r"\btorn\b",
    r"\brotura\b",
    r"\bdesgarro\b",
    r"\bruptur\b",
    r"\brupture\b",
    r"\briss\b",
    r"\bl[eé]sion\b",
    r"\blesi[oó]n\b",
    r"\bl[aä]sion\b",
    r"\bpresent\b",
    r"\bpresente\b",
    r"\beffusion\b",
    r"\bderrame\b",
    r"\b[eé]panchement\b",
    r"\berguss\b",
    r"\bedema\b",
    r"\bdegene\w+",
    r"\bartrosis\b",
    r"\bosteoarthr\w+",
    r"\bgonarthr\w+",
    r"\bsevere\b",
    r"\bmoderate\b",
    r"\bmoderado\b",
    r"\bgrave\b",
    r"\bdisruption\b",
    r"\bfractur\w+",
    r"\bcyst\b",
    r"\bquiste\b",
    r"\bkyste\b",
    r"\bzyste\b",
]

# Empirical base rates observed in gold standard
EMPIRICAL_BASE_RATES = {
    "ACL": 0.414,
    "MCL": 0.155,
    "Medial Meniscus": 0.448,
    "Lateral Meniscus": 0.397,
    "Medial OA": 0.259,
    "Lateral OA": 0.190,
    "PF OA": 0.362,
    "Effusion": 0.603,
    "Synovitis": 0.466,
    "Baker's": 0.207,
    "Contusion": 0.328,
    "Fracture": 0.310,
}


class ClinicalReportExtractor:
    def __init__(self):
        self.target_regexes = {
            t: [re.compile(p, re.IGNORECASE) for p in pats]
            for t, pats in TARGET_PATTERNS.items()
        }
        self.neg_regexes = [re.compile(p, re.IGNORECASE) for p in NEGATION_WORDS]
        self.pos_regexes = [re.compile(p, re.IGNORECASE) for p in POSITIVE_WORDS]

    def extract_from_report(self, report_text: str) -> dict:
        """
        Parses a single radiology report and assigns probabilities [0.0, 1.0] for all 12 targets.
        """
        if not isinstance(report_text, str) or not report_text.strip():
            return EMPIRICAL_BASE_RATES.copy()

        text_lower = report_text.lower()
        # Split into sentences / clauses
        clauses = re.split(r"[\n\.;:•\-\–]", text_lower)
        clauses = [c.strip() for c in clauses if c.strip()]

        scores = {}
        for target, regex_list in self.target_regexes.items():
            base_rate = EMPIRICAL_BASE_RATES[target]
            target_clauses = []

            # Check matching clauses
            for clause in clauses:
                if any(rx.search(clause) for rx in regex_list):
                    target_clauses.append(clause)

            if not target_clauses:
                # Target not explicitly mentioned -> assign baseline prior slightly shrunk
                scores[target] = base_rate * 0.7
                continue

            # Target mentioned: evaluate sentiment (negated vs positive)
            clause_scores = []
            for clause in target_clauses:
                has_negation = any(neg.search(clause) for neg in self.neg_regexes)
                has_positive = any(pos.search(clause) for pos in self.pos_regexes)

                # Special checks for OA and cysts where mention itself is often pathological
                if target in ["Medial OA", "Lateral OA", "PF OA", "Baker's", "Effusion", "Synovitis", "Contusion", "Fracture"]:
                    if has_negation:
                        clause_scores.append(0.08)
                    elif has_positive:
                        clause_scores.append(0.92)
                    else:
                        # Mentioned without explicit negation
                        clause_scores.append(0.80)
                else:
                    # Ligaments and menisci (ACL, MCL, Menisci) - often listed as normal/intact
                    if has_negation:
                        clause_scores.append(0.06)
                    elif has_positive:
                        clause_scores.append(0.88)
                    else:
                        clause_scores.append(0.35)

            # Aggregate clause scores
            if any(s > 0.5 for s in clause_scores):
                scores[target] = max(clause_scores)
            else:
                scores[target] = min(clause_scores)

        return scores

    def extract_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts predictions for all rows in a dataframe containing a 'Report' column.
        """
        records = []
        for idx, row in df.iterrows():
            scores = self.extract_from_report(row["Report"])
            scores["StudyInstanceUID"] = row["StudyInstanceUID"]
            records.append(scores)
        pred_df = pd.DataFrame(records)
        return pred_df[["StudyInstanceUID"] + TARGET_COLUMNS]


def evaluate_extractor():
    print("=" * 70)
    print("   EVALUATING CLINICAL REPORT EXTRACTOR ON 58 GOLD-STANDARD STUDIES    ")
    print("=" * 70)

    train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)
    labeled_df = train_df[train_df["ACL"].notna()].copy()

    extractor = ClinicalReportExtractor()
    preds = extractor.extract_dataset(labeled_df)

    auc_scores = {}
    print(f"\nTarget Performance on 58 Ground-Truth Studies:")
    print(f"{'Target':20s} | {'AUC':6s} | {'Positives':9s} | {'Mean Pred (Pos)':15s} | {'Mean Pred (Neg)':15s}")
    print("-" * 75)

    for target in TARGET_COLUMNS:
        y_true = labeled_df[target].values
        y_pred = preds[target].values
        try:
            auc = roc_auc_score(y_true, y_pred)
            auc_scores[target] = auc
        except Exception as e:
            auc = np.nan
            auc_scores[target] = auc

        pos_mask = y_true == 1
        mean_pos = y_pred[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        mean_neg = y_pred[~pos_mask].mean() if (~pos_mask).sum() > 0 else np.nan

        print(f"{target:20s} | {auc:6.4f} | {pos_mask.sum():2d}/{len(y_true):2d}    | {mean_pos:15.3f} | {mean_neg:15.3f}")

    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    macro_auc = np.mean(valid_aucs)
    print("-" * 75)
    print(f"OVERALL MACRO ROC-AUC (Report NLP on Gold Set): {macro_auc:.4f}")
    print("=" * 75)

    return macro_auc, preds


if __name__ == "__main__":
    evaluate_extractor()
