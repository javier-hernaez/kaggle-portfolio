"""
Exploratory Data Analysis for RSNA Knee Abnormality Detection.
Analyzes:
  1. Target label distribution & correlations in the 58 ground-truth cases.
  2. Series metadata (planes, fluid sensitivity, fat suppression) per study.
  3. Radiology reports characteristics (lengths, detected languages, terminology).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

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


def run_eda():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("      RSNA KNEE ABNORMALITY DETECTION - EXPLORATORY DATA ANALYSIS      ")
    print("=" * 70)

    train_path = DATA_DIR / "train.csv"
    train_series_path = DATA_DIR / "train_series.csv"
    test_path = DATA_DIR / "test.csv"
    test_series_path = DATA_DIR / "test_series.csv"

    train_df = pd.read_csv(train_path)
    train_series = pd.read_csv(train_series_path)
    test_df = pd.read_csv(test_path)
    test_series = pd.read_csv(test_series_path)

    # 1. Dataset Overview
    print(f"\n[1] DATASET OVERVIEW:")
    print(f"  - Total train studies: {len(train_df)}")
    labeled_mask = train_df["ACL"].notna()
    labeled_df = train_df[labeled_mask].copy()
    unlabeled_df = train_df[~labeled_mask].copy()
    print(f"  - Human ground-truth labeled studies: {len(labeled_df)} ({len(labeled_df)/len(train_df)*100:.2f}%)")
    print(f"  - Studies with unparsed reports only: {len(unlabeled_df)} ({len(unlabeled_df)/len(train_df)*100:.2f}%)")
    print(f"  - Test studies: {len(test_df)}")
    print(f"  - Train series: {len(train_series)} across {train_series['StudyInstanceUID'].nunique()} studies")
    print(f"  - Test series: {len(test_series)} across {test_series['StudyInstanceUID'].nunique()} studies")

    # 2. Target Distribution in Ground-Truth Studies
    print(f"\n[2] TARGET PREVALENCE IN 58 GOLD-STANDARD STUDIES:")
    stats_list = []
    for col in TARGET_COLUMNS:
        pos = int(labeled_df[col].sum())
        total = len(labeled_df)
        rate = pos / total
        stats_list.append({"Target": col, "Positive": pos, "Negative": total - pos, "Positive_Rate": rate})

    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False, formatters={"Positive_Rate": "{:.1%}".format}))

    # Save target prevalence table
    stats_df.to_csv(OUTPUT_DIR / "target_prevalence_58_gold.csv", index=False)

    # 3. Label Correlations
    print(f"\n[3] TARGET CORRELATION MATRIX (PEARSON, 58 GOLD-STANDARD CASES):")
    corr = labeled_df[TARGET_COLUMNS].corr()
    print(np.round(corr, 2))
    corr.to_csv(OUTPUT_DIR / "target_correlations.csv")

    # Notable correlations
    high_corrs = []
    for i in range(len(TARGET_COLUMNS)):
        for j in range(i + 1, len(TARGET_COLUMNS)):
            c1, c2 = TARGET_COLUMNS[i], TARGET_COLUMNS[j]
            val = corr.loc[c1, c2]
            if abs(val) >= 0.25:
                high_corrs.append((c1, c2, val))
    high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\n  Notable Target Co-occurrences (|r| >= 0.25):")
    for c1, c2, val in high_corrs:
        print(f"   * {c1} <--> {c2}: r = {val:+.2f}")

    # 4. Series Metadata Analysis
    print(f"\n[4] MRI SERIES METADATA ANALYSIS:")
    series_per_study = train_series.groupby("StudyInstanceUID").size()
    print(f"  - Series per study: Mean={series_per_study.mean():.2f}, Median={series_per_study.median():.0f}, Min={series_per_study.min()}, Max={series_per_study.max()}")

    plane_dist = train_series["Anatomical_Plane"].value_counts(normalize=True) * 100
    print("  - Anatomical Planes Distribution (Train):")
    for plane, pct in plane_dist.items():
        print(f"     * {plane}: {pct:.1f}%")

    plane_counts_per_study = train_series.groupby(["StudyInstanceUID", "Anatomical_Plane"]).size().unstack(fill_value=0)
    print("\n  - Mean planes per study:")
    for col in plane_counts_per_study.columns:
        print(f"     * {col}: {plane_counts_per_study[col].mean():.2f}")

    # 5. Radiology Reports Profiling
    print(f"\n[5] RADIOLOGY REPORTS PROFILING:")
    train_df["report_chars"] = train_df["Report"].str.len()
    train_df["report_words"] = train_df["Report"].str.split().str.len()
    print(f"  - Character length: Min={train_df['report_chars'].min()}, Max={train_df['report_chars'].max()}, Median={train_df['report_chars'].median():.0f}")
    print(f"  - Word count: Min={train_df['report_words'].min()}, Max={train_df['report_words'].max()}, Median={train_df['report_words'].median():.0f}")

    # Check languages by common medical stop/key terms
    sample_terms = {
        "Spanish": ["rodilla", "menisco", "derrame", "ligamento", "edema"],
        "English": ["knee", "meniscus", "effusion", "ligament", "tear"],
        "German": ["knie", "meniskus", "erguss", "band", "ruptur"],
        "French": ["genou", "ménisque", "épanchement", "ligament", "déchirure"],
        "Portuguese": ["joelho", "menisco", "derrame", "ligamento", "edema"],
    }
    lang_matches = {lang: 0 for lang in sample_terms}
    for report in train_df["Report"]:
        rep_lower = str(report).lower()
        for lang, words in sample_terms.items():
            if any(w in rep_lower for w in words):
                lang_matches[lang] += 1

    print("\n  - Detected Language Lexicon Presence (approximate counts across 4,407 reports):")
    for lang, cnt in sorted(lang_matches.items(), key=lambda x: x[1], reverse=True):
        print(f"     * {lang:12s}: {cnt:4d} reports ({cnt/len(train_df)*100:.1f}%)")

    print("\n[+] EDA complete. Artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
