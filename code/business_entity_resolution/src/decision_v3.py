"""
Vectorized decision layer + metric, shared by training (OOF tuning) and inference.

Exclusivity: invariant I1 (verified: 0 duplicate links in train GT) says each S2/S3 record
belongs to at most one S1. Greedy global exclusivity keeps a candidate only for the S1 where
its calibrated p is highest. Because every row of a given candidate shares the same source and
country, it also shares the same threshold, so the winning S1 does not depend on the threshold:
the winner is simply the highest-p row of that candidate. That makes a full threshold grid a
set of cheap masks instead of repeated sorts.
"""

import numpy as np


def exclusive_best_mask(p: np.ndarray, cid_codes: np.ndarray) -> np.ndarray:
    """True for the highest-p row of each candidate id (ties broken by row order)."""
    order = np.lexsort((np.arange(len(p)), -p))  # p desc, stable
    best = np.zeros(len(p), dtype=bool)
    c_sorted = cid_codes[order]
    # first occurrence of each cid in sorted order
    _, first_idx = np.unique(c_sorted, return_index=True)
    best[order[first_idx]] = True
    return best


def row_thresholds(is_s3: np.ndarray, th_s2: float, th_s3: float,
                   unseen_mask: np.ndarray = None, unseen_offset: float = 0.0) -> np.ndarray:
    th = np.where(is_s3.astype(bool), th_s3, th_s2).astype(np.float32)
    if unseen_mask is not None and unseen_offset:
        th = th + unseen_mask.astype(np.float32) * unseen_offset
    return th


def f05_per_s1(sid_codes: np.ndarray, pred: np.ndarray, label: np.ndarray,
               ntrue: np.ndarray) -> np.ndarray:
    """
    Per-S1 F0.5 exactly per the challenge rules, for ALL S1 (len(ntrue)), including S1 with no
    candidates. ntrue counts GT links even when blocking missed them.
    """
    n = len(ntrue)
    npred = np.bincount(sid_codes, weights=pred, minlength=n)
    tp = np.bincount(sid_codes, weights=pred & label, minlength=n)
    f = np.zeros(n, dtype=np.float64)
    single = ntrue == 0
    f[single] = (npred[single] == 0).astype(np.float64)
    m = (~single) & (npred > 0) & (tp > 0)
    prec = tp[m] / npred[m]
    rec = tp[m] / ntrue[m]
    f[m] = 1.25 * prec * rec / (0.25 * prec + rec)
    return f


def summarize(sid_codes, pred, label, ntrue, s1_country=None) -> dict:
    f = f05_per_s1(sid_codes, pred, label, ntrue)
    n = len(ntrue)
    npred = np.bincount(sid_codes, weights=pred, minlength=n)
    tp = np.bincount(sid_codes, weights=pred & label, minlength=n)
    out = {
        "macro_f05": float(f.mean()),
        "micro_p": float(tp.sum() / max(1, npred.sum())),
        "micro_r": float(tp.sum() / max(1, ntrue.sum())),
        "singleton_acc": float(f[ntrue == 0].mean()) if (ntrue == 0).any() else None,
        "mean_pred_per_s1": float(npred.mean()),
        "k_buckets": {},
    }
    for k in range(0, 7):
        mk = (ntrue == k) if k < 6 else (ntrue >= 6)
        if mk.any():
            out["k_buckets"][str(k) if k < 6 else "6+"] = round(float(f[mk].mean()), 4)
    if s1_country is not None:
        out["country_f05"] = {str(c): round(float(f[s1_country == c].mean()), 4)
                              for c in np.unique(s1_country)}
    return out
