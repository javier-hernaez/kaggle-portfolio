"""
Ingeniería de variables para NLP with Disaster Tweets.
Genera meta-features textuales, combina keywords y prepara
matrices TF-IDF (palabras + caracteres) y variables tabulares.
"""

import re
import string
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from text_cleaner import clean_tweet_text, clean_keyword

def extract_meta_features(df):
    """Extrae métricas numéricas y estilométricas del texto original antes de la limpieza."""
    meta = pd.DataFrame(index=df.index)

    raw_texts = df['text'].fillna("").astype(str)
    
    # Longitud y conteo básico
    meta['char_count'] = raw_texts.str.len()
    meta['word_count'] = raw_texts.apply(lambda x: len(x.split()))
    meta['unique_word_count'] = raw_texts.apply(lambda x: len(set(x.split())))
    meta['mean_word_length'] = meta['char_count'] / (meta['word_count'] + 1e-5)
    
    # Mayúsculas (gritos de alerta o énfasis)
    meta['uppercase_count'] = raw_texts.apply(lambda x: sum(1 for c in x if c.isupper()))
    meta['caps_ratio'] = meta['uppercase_count'] / (meta['char_count'] + 1e-5)

    # Signos y puntuación
    meta['punctuation_count'] = raw_texts.apply(lambda x: sum(1 for c in x if c in string.punctuation))
    meta['exclamation_count'] = raw_texts.apply(lambda x: x.count('!'))
    meta['question_count'] = raw_texts.apply(lambda x: x.count('?'))

    # Elementos de Twitter
    meta['url_count'] = raw_texts.apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    meta['mention_count'] = raw_texts.apply(lambda x: len(re.findall(r'@\w+', x)))
    meta['hashtag_count'] = raw_texts.apply(lambda x: len(re.findall(r'#\w+', x)))

    # Metadatos de keyword y location
    meta['has_keyword'] = df['keyword'].notna().astype(int)
    meta['has_location'] = df['location'].notna().astype(int)

    # Coincidencia de la palabra clave en el texto
    def check_kw_in_text(row):
        kw = str(row['keyword'])
        if pd.isna(row['keyword']) or kw == "missing":
            return 0
        cleaned_kw = clean_keyword(kw)
        return int(cleaned_kw in str(row['text']).lower())

    meta['kw_in_text'] = df.apply(check_kw_in_text, axis=1)

    return meta

def prepare_text_series(df):
    """
    Genera el texto enriquecido combinando la keyword limpia y el texto normalizado.
    La keyword actúa como prefijo temático de alta relevancia.
    """
    clean_kws = df['keyword'].apply(clean_keyword)
    clean_texts = df['text'].apply(clean_tweet_text)

    # Si hay keyword, la prependeamos al texto: "earthquake: the building shook violently"
    combined = clean_kws.where(clean_kws != "missing", "") + " " + clean_texts
    return clean_texts, combined.str.strip()

class TextFeaturePipeline:
    """Pipeline que combina TF-IDF a nivel de palabra, carácter y meta-features numéricas."""
    def __init__(self, max_word_features=12000, max_char_features=15000):
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_word_features,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        self.char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer='char',
            max_features=max_char_features,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.scaler = StandardScaler()
        self.meta_cols = None

    def fit_transform(self, df):
        _, combined_texts = prepare_text_series(df)
        
        # 1. TF-IDF Palabras
        X_word = self.word_vec.fit_transform(combined_texts)
        
        # 2. TF-IDF Caracteres
        X_char = self.char_vec.fit_transform(combined_texts)
        
        # 3. Meta-features escaladas
        meta_df = extract_meta_features(df)
        self.meta_cols = meta_df.columns
        X_meta_scaled = self.scaler.fit_transform(meta_df)
        X_meta_sparse = csr_matrix(X_meta_scaled)

        # 4. Concatenación dispersa
        X_all = hstack([X_word, X_char, X_meta_sparse], format='csr')
        return X_all

    def transform(self, df):
        _, combined_texts = prepare_text_series(df)
        
        X_word = self.word_vec.transform(combined_texts)
        X_char = self.char_vec.transform(combined_texts)
        
        meta_df = extract_meta_features(df)
        X_meta_scaled = self.scaler.transform(meta_df[self.meta_cols])
        X_meta_sparse = csr_matrix(X_meta_scaled)

        X_all = hstack([X_word, X_char, X_meta_sparse], format='csr')
        return X_all
