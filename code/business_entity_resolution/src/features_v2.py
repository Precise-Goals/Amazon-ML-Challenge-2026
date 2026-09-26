import re
import numpy as np
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

def extract_nums(addr: str) -> Set[str]:
    raw = re.findall(r'\b\d+\b', addr or "")
    return {str(int(n)) for n in raw if len(n) <= 5}

def compute_group_context_features(
    cand_list: List[dict]
) -> List[List[float]]:
    """
    Computes group-context and candidate-agreement features within one S1's candidate set.
    cand_list: list of dict with keys:
       'cid', 'comb_sim', 'n_tsort', 'n_tset', 'a_tset', 'brand', 'addr', 'src' ('S2' or 'S3')
    """
    n_cands = len(cand_list)
    if n_cands == 0:
        return []

    comb_sims = np.array([c['comb_sim'] for c in cand_list], dtype=np.float32)
    name_sims = np.array([c['n_tsort'] for c in cand_list], dtype=np.float32)

    best_comb = float(np.max(comb_sims))
    best_name = float(np.max(name_sims))
    mean_comb = float(np.mean(comb_sims))
    std_comb = float(np.std(comb_sims)) + 1e-5

    # Ranks (0 = highest score)
    comb_order = np.argsort(-comb_sims)
    rank_comb = np.empty(n_cands, dtype=np.float32)
    rank_comb[comb_order] = np.arange(n_cands)

    name_order = np.argsort(-name_sims)
    rank_name = np.empty(n_cands, dtype=np.float32)
    rank_name[name_order] = np.arange(n_cands)

    # Source counts and source-internal ranks
    s2_indices = [i for i, c in enumerate(cand_list) if c['src'] == 'S2']
    s3_indices = [i for i, c in enumerate(cand_list) if c['src'] == 'S3']
    n_s2 = float(len(s2_indices))
    n_s3 = float(len(s3_indices))

    rank_within_src = np.zeros(n_cands, dtype=np.float32)
    if s2_indices:
        s2_sims = comb_sims[s2_indices]
        s2_order = np.argsort(-s2_sims)
        for r, idx in enumerate(np.array(s2_indices)[s2_order]):
            rank_within_src[idx] = float(r)
    if s3_indices:
        s3_sims = comb_sims[s3_indices]
        s3_order = np.argsort(-s3_sims)
        for r, idx in enumerate(np.array(s3_indices)[s3_order]):
            rank_within_src[idx] = float(r)

    # Candidate agreement features (D7)
    # Pre-extract brand strings
    brands = [c.get('brand', '') for c in cand_list]
    agree_max_other = np.zeros(n_cands, dtype=np.float32)
    agree_mean_other = np.zeros(n_cands, dtype=np.float32)
    count_sim_gt_90 = np.zeros(n_cands, dtype=np.float32)

    for i in range(n_cands):
        this_src = cand_list[i]['src']
        other_indices = s3_indices if this_src == 'S2' else s2_indices
        
        sims_to_other = []
        for j in other_indices:
            if brands[i] and brands[j]:
                sims_to_other.append(fuzz.token_set_ratio(brands[i], brands[j]) / 100.0)
            else:
                sims_to_other.append(0.0)
                
        if sims_to_other:
            agree_max_other[i] = float(np.max(sims_to_other))
            agree_mean_other[i] = float(np.mean(sims_to_other))
            
        # Count other candidates with brand sim > 0.90
        gt90 = 0
        for j in range(n_cands):
            if i != j and brands[i] and brands[j]:
                if fuzz.token_sort_ratio(brands[i], brands[j]) >= 90:
                    gt90 += 1
        count_sim_gt_90[i] = float(gt90)

    # Assemble group feature rows
    group_feats = []
    for i in range(n_cands):
        c_sim = comb_sims[i]
        n_sim = name_sims[i]
        row = [
            rank_comb[i],
            rank_name[i],
            best_comb - c_sim,
            best_name - n_sim,
            (c_sim - mean_comb) / std_comb,
            float(n_cands),
            n_s2,
            n_s3,
            rank_within_src[i],
            agree_max_other[i],
            agree_mean_other[i],
            count_sim_gt_90[i]
        ]
        group_feats.append(row)

    return group_feats

def compute_reverse_context_features(
    all_s1_pairs: Dict[str, List[dict]]
) -> Dict[Tuple[str, str], List[float]]:
    """
    Computes reverse context across all pairs:
    cand_s1_count: how many S1s retrieved this candidate
    is_mutual_best: 1.0 if this S1 gave this candidate its highest score
    gap_to_second_s1: difference between this S1 score and the 2nd-best S1 score
    """
    # Mapping: cid -> list of (sid, score)
    cand_to_s1s = {}
    for sid, items in all_s1_pairs.items():
        for item in items:
            cid = item['cid']
            sc = item['comb_sim']
            if cid not in cand_to_s1s:
                cand_to_s1s[cid] = []
            cand_to_s1s[cid].append((sid, sc))

    reverse_map = {}
    for cid, s1_list in cand_to_s1s.items():
        cnt = len(s1_list)
        if cnt == 1:
            sid, sc = s1_list[0]
            reverse_map[(sid, cid)] = [1.0, 1.0, 1.0]
        else:
            # Sort descending by score
            s1_list.sort(key=lambda x: x[1], reverse=True)
            best_sid, best_sc = s1_list[0]
            second_sc = s1_list[1][1]
            
            for rank, (sid, sc) in enumerate(s1_list):
                if rank == 0:
                    is_mut = 1.0
                    gap = best_sc - second_sc
                else:
                    is_mut = 0.0
                    gap = sc - best_sc  # Negative gap
                reverse_map[(sid, cid)] = [float(cnt), is_mut, float(gap)]

    return reverse_map
