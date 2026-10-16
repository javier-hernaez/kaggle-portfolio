"""
Feature Engineering for Name That LLM.
Extracts rich stylometric, punctuation, syntactic, and n-gram representations.
"""

import re
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Canonical LLM buzzwords and discourse markers
LLM_MARKERS = [
    "however", "moreover", "furthermore", "crucially", "ultimately", "indeed",
    "thus", "therefore", "notably", "specifically", "essentially", "delve",
    "tapestry", "testament", "beacon", "realm", "intricate", "nuanced",
    "underscores", "underscoring", "fosters", "pivotal", "in essence",
    "at its core", "it is worth noting", "serves as", "a testament to"
]

STOPWORDS_LIST = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts dense stylistic and syntactic features from text.
    Designed to capture authoring LLM fingerprints (em-dashes, quote styles, contractions, etc.).
    """

    def __init__(self, scale=True):
        self.scale = scale
        self.scaler = StandardScaler()
        self.feature_names_ = []

    def fit(self, X, y=None):
        df_feats = self._extract(X)
        self.feature_names_ = df_feats.columns.tolist()
        if self.scale:
            self.scaler.fit(df_feats.values)
        return self

    def transform(self, X):
        df_feats = self._extract(X)
        vals = df_feats.values
        if self.scale:
            vals = self.scaler.transform(vals)
        return vals

    def _extract(self, texts):
        records = []
        for text in texts:
            if not isinstance(text, str):
                text = ""

            char_len = len(text)
            words = text.split()
            word_len = max(len(words), 1)

            # Length & structure
            avg_word_len = char_len / word_len
            word_lens = [len(w) for w in words] if words else [0]
            word_len_std = np.std(word_lens) if len(word_lens) > 1 else 0.0

            # Sentences
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            sent_count = max(len(sentences), 1)
            sent_lens = [len(s.split()) for s in sentences] if sentences else [word_len]
            avg_sent_len = np.mean(sent_lens)
            sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0.0

            # Punctuation counts
            em_dash_count = text.count("—") + text.count("--")
            colon_count = text.count(":")
            semicolon_count = text.count(";")
            comma_count = text.count(",")
            period_count = text.count(".")
            question_count = text.count("?")
            exclamation_count = text.count("!")
            ellipsis_count = text.count("...") + text.count("…")
            parentheses_count = text.count("(") + text.count(")")
            hyphen_count = text.count("-") - (text.count("--") * 2)
            hyphen_count = max(0, hyphen_count)

            # Quotations (Straight vs Curly)
            straight_double = text.count('"')
            straight_single = text.count("'")
            curly_double = text.count("“") + text.count("”")
            curly_single = text.count("‘") + text.count("’")
            total_straight = straight_double + straight_single
            total_curly = curly_double + curly_single

            # Markdown / Formats
            newlines = text.count("\n")
            bold_stars = text.count("**")
            asterisks = text.count("*") - (bold_stars * 2)
            bullet_points = len(re.findall(r"(?:^|\n)\s*[-*•]\s", text))

            # Contractions & Casing
            contractions = len(re.findall(r"\b\w+(?:'|’)(?:t|s|ve|re|ll|d|m)\b", text, re.IGNORECASE))
            capitals = sum(1 for c in text if c.isupper())
            caps_ratio = capitals / max(char_len, 1)
            all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)

            # Lexical diversity & Stopwords
            words_lower = [w.lower() for w in words]
            unique_words = len(set(words_lower))
            ttr = unique_words / word_len
            stopwords_count = sum(1 for w in words_lower if w in STOPWORDS_LIST)
            stopword_ratio = stopwords_count / word_len

            # Discourse markers count
            text_lower = text.lower()
            marker_counts = sum(text_lower.count(m) for m in LLM_MARKERS)

            # Normalized densities per 100 words
            factor = 100.0 / word_len

            record = {
                # Raw and structural lengths
                "char_len": char_len,
                "word_len": word_len,
                "avg_word_len": avg_word_len,
                "word_len_std": word_len_std,
                "sent_count": sent_count,
                "avg_sent_len": avg_sent_len,
                "sent_len_std": sent_len_std,
                "newlines": newlines,
                "caps_ratio": caps_ratio,
                "all_caps_words": all_caps_words,
                "ttr": ttr,
                "stopword_ratio": stopword_ratio,
                # Stylistic Punctuation Counts
                "em_dash_count": em_dash_count,
                "colon_count": colon_count,
                "semicolon_count": semicolon_count,
                "comma_count": comma_count,
                "period_count": period_count,
                "question_count": question_count,
                "exclamation_count": exclamation_count,
                "ellipsis_count": ellipsis_count,
                "parentheses_count": parentheses_count,
                "hyphen_count": hyphen_count,
                # Quote counts & features
                "straight_double": straight_double,
                "straight_single": straight_single,
                "curly_double": curly_double,
                "curly_single": curly_single,
                "total_curly": total_curly,
                "has_curly_quotes": float(total_curly > 0),
                "has_em_dash": float(em_dash_count > 0),
                # Normalized Punctuation Densities (per 100 words)
                "em_dash_per_100": em_dash_count * factor,
                "colon_per_100": colon_count * factor,
                "semicolon_per_100": semicolon_count * factor,
                "comma_per_100": comma_count * factor,
                "contractions_per_100": contractions * factor,
                "markers_per_100": marker_counts * factor,
                # Formatting
                "bold_stars": bold_stars,
                "asterisks": max(0, asterisks),
                "bullet_points": bullet_points,
            }
            records.append(record)

        return pd.DataFrame(records)


def build_text_pipelines(
    max_features_word: int = 15000,
    max_features_char: int = 40000,
    char_ngram_range: tuple = (3, 6),
    word_ngram_range: tuple = (1, 2),
):
    """
    Constructs the TF-IDF feature pipelines:
    1. Word-level n-grams (preserves stopwords & case for stylistic signal)
    2. Character-level n-grams (captures exact punctuation sequences, suffixes, formatting)
    """
    word_vec = TfidfVectorizer(
        ngram_range=word_ngram_range,
        max_features=max_features_word,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
    )

    char_vec = TfidfVectorizer(
        ngram_range=char_ngram_range,
        analyzer="char",
        max_features=max_features_char,
        sublinear_tf=True,
        min_df=3,
    )

    return word_vec, char_vec


def extract_all_features(
    train_texts,
    test_texts,
    max_features_word: int = 15000,
    max_features_char: int = 40000,
):
    """
    Fits and extracts full feature representation for both train and test.
    Returns:
        X_train_sparse, X_test_sparse, feature_names
    """
    print("[*] Extracting stylometric features...")
    sty = StylometricExtractor(scale=True)
    sty_train = sty.fit_transform(train_texts)
    sty_test = sty.transform(test_texts)

    print(f"[*] Fitting Word TF-IDF (max_features={max_features_word})...")
    word_vec, char_vec = build_text_pipelines(
        max_features_word=max_features_word,
        max_features_char=max_features_char,
    )
    w_train = word_vec.fit_transform(train_texts)
    w_test = word_vec.transform(test_texts)

    print(f"[*] Fitting Character TF-IDF (max_features={max_features_char})...")
    c_train = char_vec.fit_transform(train_texts)
    c_test = char_vec.transform(test_texts)

    print("[*] Stacking stylometric + word + char sparse representations...")
    X_train = hstack([csr_matrix(sty_train), w_train, c_train]).tocsr()
    X_test = hstack([csr_matrix(sty_test), w_test, c_test]).tocsr()

    print(f"[✔] Total features extracted: {X_train.shape[1]:,} | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, sty, word_vec, char_vec
