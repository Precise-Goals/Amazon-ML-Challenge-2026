"""
Feature extraction pipeline for candidate pairs using RapidFuzz string metrics.
"""

import re
from typing import List, Tuple, Set, Dict, Any
import numpy as np
from rapidfuzz import fuzz
try:
    from .utils import normalize_text, get_address_components
except (ImportError, ValueError):
    from utils import normalize_text, get_address_components



FEATURE_NAMES = [
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_wratio',
    'addr_ratio',
    'addr_partial_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'max_sim',
    'comb_sim',
    'dual_high',
    'num_overlap_ratio',
    'exact_num_match',
    'num_match',
    'zip_match',
    'len_diff_ratio',
    'is_s2',
    'is_s3'
]

def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_nums: Set[str],
    s1_zip: str,
    c_id: str,
    c_name: str,
    c_addr: str,
    c_nums: Set[str],
    c_zip: str,
    precomputed_name_tset: float = None,
    precomputed_addr_tset: float = None
) -> List[float]:
    """Compute dense feature vector for a pair (Source 1, Candidate)."""
    # Name string metrics
    n_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    n_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    n_tsort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    n_tset = precomputed_name_tset if precomputed_name_tset is not None else fuzz.token_set_ratio(s1_name, c_name) / 100.0
    n_wratio = fuzz.WRatio(s1_name, c_name) / 100.0
    
    # Address string metrics
    if s1_addr and c_addr and s1_addr != 'nan' and c_addr != 'nan':
        a_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
        a_partial = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
        a_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        a_tset = precomputed_addr_tset if precomputed_addr_tset is not None else fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    else:
        a_ratio = 0.0
        a_partial = 0.0
        a_tsort = 0.0
        a_tset = 0.0
        
    # Interaction metrics
    max_sim = max(n_tset, a_tset)
    comb_sim = 0.6 * n_tset + 0.4 * a_tset
    dual_high = 1.0 if (n_tset >= 0.6 and a_tset >= 0.5) else 0.0
        
    # Numbers and postal codes
    if s1_nums and c_nums:
        num_overlap = len(s1_nums.intersection(c_nums)) / max(len(s1_nums), len(c_nums), 1)
        exact_num_match = 1.0 if s1_nums == c_nums else 0.0
        num_match = 1.0 if len(s1_nums.intersection(c_nums)) > 0 else 0.0
    else:
        num_overlap = 0.0
        exact_num_match = 0.0
        num_match = 0.0
        
    zip_match = 1.0 if (s1_zip and c_zip and s1_zip == c_zip) else 0.0
    
    # Length difference
    max_len = max(len(s1_name), len(c_name), 1)
    len_diff_ratio = abs(len(s1_name) - len(c_name)) / max_len
    
    # Source indicator
    is_s2 = 1.0 if c_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if c_id.startswith('S3-') else 0.0
    
    return [
        n_ratio,
        n_partial,
        n_tsort,
        n_tset,
        n_wratio,
        a_ratio,
        a_partial,
        a_tsort,
        a_tset,
        max_sim,
        comb_sim,
        dual_high,
        num_overlap,
        exact_num_match,
        num_match,
        zip_match,
        len_diff_ratio,
        is_s2,
        is_s3
    ]
