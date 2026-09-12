"""
Módulo de limpieza y normalización de texto para Tweets de desastre.
Gestiona entidades HTML, URLs, menciones, hashtags, contracciones
y caracteres mojibake específicos de este dataset de Twitter.
"""

import re
import html
import urllib.parse

# Mapeo de contracciones frecuentes en inglés
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "im": "i am",
    "u": "you",
    "r": "are",
    "ur": "your",
    "dont": "do not",
    "didnt": "did not",
    "doesnt": "does not",
    "wont": "will not",
    "cant": "cannot",
}

# Artefactos típicos de codificación (mojibake) en el dataset de Kaggle
MOJIBAKE_MAP = {
    "\x89Û_": "...",
    "\x89Ûª": "'",
    "\x89Û÷": "'",
    "\x89ÛÒ": "-",
    "\x89ÛÓ": "-",
    "\x89ÛÏ": "\"",
    "\x89Û\x9d": "\"",
    "\x89Û": " ",
    "&amp;": " and ",
    "&lt;": " < ",
    "&gt;": " > ",
    "&quot;": " \" ",
}

def clean_keyword(kw):
    """Limpia el campo keyword decodificando URLs y eliminando espacios superfluos."""
    if pd_isna(kw):
        return "missing"
    kw = urllib.parse.unquote(str(kw))
    kw = kw.replace("%20", " ").strip().lower()
    return kw

def pd_isna(val):
    if val is None:
        return True
    if isinstance(val, float) and val != val:
        return True
    s = str(val).strip().lower()
    return s in ["", "nan", "none", "null"]

def clean_tweet_text(text, keep_hashtags=True):
    """
    Limpieza y normalización de un tweet:
    1. Decodificar entidades HTML.
    2. Corregir mojibake conocido.
    3. Reemplazar URLs por token ' http_link '.
    4. Reemplazar menciones por token ' user_mention '.
    5. Desempaquetar hashtags (#wildfire -> wildfire).
    6. Expandir contracciones comunes.
    7. Eliminar puntuación excesiva y espacios repetidos.
    """
    if pd_isna(text):
        return ""

    text = str(text)

    # 1. HTML entities
    text = html.unescape(text)

    # 2. Mojibake
    for bad_str, rep in MOJIBAKE_MAP.items():
        text = text.replace(bad_str, rep)

    # 3. URLs -> token
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)

    # 4. Menciones -> token
    text = re.sub(r'@\w+', ' mention ', text)

    # 5. Hashtags
    if keep_hashtags:
        # Extraer el término tras la almohadilla
        text = re.sub(r'#(\w+)', r' \1 ', text)
    else:
        text = re.sub(r'#\w+', ' ', text)

    # 6. Minúsculas
    text = text.lower()

    # 7. Expandir contracciones
    for cont, exp in CONTRACTIONS.items():
        text = re.sub(r'\b' + cont + r'\b', exp, text)

    # 8. Eliminar caracteres especiales no alfanuméricos preservando puntuación básica
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)

    # 9. Espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text
