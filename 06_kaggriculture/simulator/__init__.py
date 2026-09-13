"""
Simulator package for Kaggriculture.
Enables fast local evaluation and matches without external dependencies.
"""
from .engine import KaggricultureEnv, CROPS, ANIMALS, PRODUCTS, SHOPS, MARKET_PARAMS
from .battle import run_match, tournament

__all__ = [
    "KaggricultureEnv",
    "CROPS",
    "ANIMALS",
    "PRODUCTS",
    "SHOPS",
    "MARKET_PARAMS",
    "run_match",
    "tournament",
]
