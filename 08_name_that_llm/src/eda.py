"""
Exploratory Data Analysis and Stylometric Fingerprinting for Name That LLM.
Analyzes stylistic habits, punctuation patterns, and length distributions across 18 LLMs.
"""

import os
import re
import sys
import pandas as pd
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def get_model_family(model_name: str) -> str:
    """Map individual model name to its parent lab / architecture family."""
    if model_name.startswith("gpt"):
        return "OpenAI"
    elif model_name.startswith("claude"):
        return "Anthropic"
    elif model_name.startswith("gemini"):
        return "Google"
    elif model_name.startswith("qwen"):
        return "Alibaba"
    elif model_name.startswith("glm"):
        return "Zhipu"
    elif model_name.startswith("kimi"):
        return "Moonshot"
    return "Other"


def compute_stylometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract interpretable stylometric metrics from text passages."""
    feats = pd.DataFrame(index=df.index)
    
    # Length metrics
    feats["char_len"] = df["text"].str.len()
    feats["word_len"] = df["text"].apply(lambda x: len(x.split()))
    feats["avg_word_len"] = feats["char_len"] / feats["word_len"].clip(lower=1)
    feats["sentence_count"] = df["text"].apply(lambda x: len(re.split(r"[.!?]+", x)) - 1).clip(lower=1)
    feats["avg_sentence_len"] = feats["word_len"] / feats["sentence_count"]

    # Punctuation counts
    feats["em_dash"] = df["text"].apply(lambda x: x.count("—") + x.count("--"))
    feats["semicolon"] = df["text"].apply(lambda x: x.count(";"))
    feats["colon"] = df["text"].apply(lambda x: x.count(":"))
    feats["comma"] = df["text"].apply(lambda x: x.count(","))
    feats["question_mark"] = df["text"].apply(lambda x: x.count("?"))
    feats["exclamation"] = df["text"].apply(lambda x: x.count("!"))
    feats["ellipsis"] = df["text"].apply(lambda x: x.count("...") + x.count("…"))

    # Quotes: Straight vs Curly
    feats["straight_quotes"] = df["text"].apply(lambda x: x.count('"') + x.count("'"))
    feats["curly_quotes"] = df["text"].apply(lambda x: sum(x.count(q) for q in ["“", "”", "‘", "’"]))

    # Formatting & Markdown
    feats["bold_stars"] = df["text"].apply(lambda x: x.count("**"))
    feats["bullet_points"] = df["text"].apply(lambda x: len(re.findall(r"(?:^|\n)\s*[-*•]\s", x)))
    feats["newlines"] = df["text"].apply(lambda x: x.count("\n"))

    # Contractions
    feats["contractions"] = df["text"].apply(
        lambda x: len(re.findall(r"\b\w+(?:'|’)(?:t|s|ve|re|ll|d|m)\b", x, re.IGNORECASE))
    )

    # Lexical Diversity: Type-Token Ratio (TTR)
    feats["ttr"] = df["text"].apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )

    return feats


def run_eda():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        print(f"[!] train.csv not found at {train_path}. Run download_data.py first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("=" * 80)
    print(" 🔎 NAME THAT LLM - EXPLORATORY DATA ANALYSIS & STYLOMETRIC AUDIT")
    print("=" * 80)
    print(f"[*] Train dataset shape: {train_df.shape}")
    print(f"[*] Unique prompt IDs: {train_df['prompt_id'].nunique()}")
    print(f"[*] Unique models: {train_df['model'].nunique()}")
    print(f"[*] Test dataset shape: {test_df.shape}")
    print(f"[*] Models list:\n{sorted(train_df['model'].unique())}\n")

    train_df["family"] = train_df["model"].apply(get_model_family)
    stylometrics = compute_stylometric_features(train_df)
    combined = pd.concat([train_df[["model", "family", "prompt_id"]], stylometrics], axis=1)

    print("-" * 80)
    print(" 📊 STYLOMETRIC SIGNATURES BY MODEL FAMILY (Means)")
    print("-" * 80)
    family_summary = combined.groupby("family")[
        ["char_len", "word_len", "em_dash", "colon", "semicolon", "curly_quotes", "contractions", "ttr"]
    ].mean().round(3)
    print(family_summary.to_string())

    print("\n" + "-" * 80)
    print(" 🔍 DETAILED STYLOMETRIC SIGNATURES PER MODEL (Means)")
    print("-" * 80)
    model_summary = combined.groupby("model")[
        ["word_len", "em_dash", "colon", "semicolon", "curly_quotes", "contractions", "ttr"]
    ].mean().round(3)
    print(model_summary.to_string())

    print("\n" + "-" * 80)
    print(" 💡 CRITICAL STYLOMETRIC INSIGHTS FOR MODELING")
    print("-" * 80)
    print(" 1. EM-DASH (`—`):")
    print("    - Gemini models strictly avoid em-dashes (mean ≈ 0.00-0.01).")
    print("    - Claude and GLM models are heavy em-dash users (mean 0.97 - 1.62 per response).")
    print("    - GPT models use moderate em-dashes (mean 0.11 - 0.49).")
    print(" 2. COLON (`:`):")
    print("    - Gemini models rarely use colons (0.17 - 0.23).")
    print("    - GPT and Kimi models use colons 4x-5x more frequently (0.89 - 0.99).")
    print(" 3. CURLY VS STRAIGHT QUOTES:")
    print("    - Certain models output unicode curly quotes while others output ASCII straight quotes.")
    print(" 4. VALIDATION INTEGRITY:")
    print("    - GroupKFold on `prompt_id` is mandatory to avoid memorizing topic vocabulary.")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
