"""
Utility functions for text normalization, tokenization, I/O, and evaluation metrics.
"""

import re
import sys
import unicodedata
import numpy as np
import pandas as pd
from typing import List, Tuple, Set, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Regex for word tokens (preserving unicode characters)
TOKEN_RE = re.compile(r'[^\w\s]', re.UNICODE)
NUMBERS_RE = re.compile(r'\b\d+\b')
DOMAIN_RE = re.compile(r'\.(com|org|net|in|co|io|fr|gov|edu)\b', re.IGNORECASE)
PREFIX_RE = re.compile(r'(@|https?://|www\.)', re.IGNORECASE)

def strip_accents(s: str) -> str:
    """Strip combining diacritical marks/accents while preserving base characters."""
    if not s:
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_text(text: Any, clean_domains: bool = True) -> str:
    """Normalize text by lowercasing, stripping accents, and standardizing whitespace."""
    if text is None or pd.isna(text):
        return ""
    s = strip_accents(str(text).strip().lower())
    if s in ("nan", "null", "none"):
        return ""
    if clean_domains:
        s = PREFIX_RE.sub(' ', s)
        s = DOMAIN_RE.sub(' ', s)
    return ' '.join(s.split())

def get_name_tokens(name: str, stop_words: Set[str], min_len: int = 3) -> Tuple[List[str], List[str]]:
    """
    Extract informative tokens and short 2-character acronym tokens from business name.
    Returns: (standard_tokens, short_tokens)
    """
    clean = TOKEN_RE.sub(' ', name.lower())
    words = clean.split()
    tokens = [t for t in words if len(t) >= min_len and t not in stop_words]
    short_tokens = [t for t in words if len(t) == 2 and t not in stop_words]
    return tokens, short_tokens

def get_address_components(address: str, stop_words: Set[str] = None) -> Tuple[List[str], str, List[str], List[str]]:
    """
    Extract numbers, postal code, street numbers, and informative street words from business address.
    Returns: (all_numbers, zip_code, street_numbers, street_words)
    """
    clean_addr = TOKEN_RE.sub(' ', address.lower())
    nums = NUMBERS_RE.findall(address)
    
    # 5 or 6 digit postal code
    zips = [n for n in nums if len(n) in (5, 6)]
    zip_code = zips[0] if zips else ""
    street_nums = [n for n in nums if n != zip_code and len(n) <= 5]
    
    stop_set = stop_words if stop_words is not None else set()
    words = [w for w in clean_addr.split() if len(w) >= 3 and not w.isdigit() and w not in stop_set]
    return nums, zip_code, street_nums, words

def calculate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, List[str]]
) -> float:
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities.
    Formula:
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Singletons:
        - True empty, predicted empty => 1.0
        - True empty, predicted non-empty => 0.0
        - True non-empty, predicted empty => 0.0
    """
    f05_scores = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = set(predictions.get(s1_id, []))
        
        if len(true_set) == 0 and len(pred_set) == 0:
            f05_scores.append(1.0)
            continue
        if len(true_set) == 0 and len(pred_set) > 0:
            f05_scores.append(0.0)
            continue
        if len(true_set) > 0 and len(pred_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denominator = (0.25 * p + r)
        if denominator == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * p * r) / denominator
            
        f05_scores.append(f05)
        
    return float(np.mean(f05_scores)) if f05_scores else 0.0
