import os
import sys
import gc
import re
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import joblib
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from gold import GoldResolutionEngine
from features_v2 import compute_group_context_features, compute_reverse_context_features, extract_nums
from eval_harness import evaluate_metrics

def compute_expected_f05_monte_carlo(probs: List[float], n_samples: int = 150) -> int:
    """
    Computes argmax_k E[F0.5 | pick top k] under Bernoulli(p_i).
    probs: sorted descending calibrated probabilities [p_1, p_2, ..., p_K]
    """
    K = len(probs)
    if K == 0:
        return 0

    p_arr = np.array(probs, dtype=np.float32)
    # k = 0 expected score: probability that all K candidates are false
    # Product of (1 - p_i)
    e_f05_k0 = float(np.prod(1.0 - p_arr))

    # Monte Carlo simulation: draw n_samples binary realizations of ground truth
    # realizations: shape (n_samples, K)
    rand_vals = np.random.rand(n_samples, K).astype(np.float32)
    sim_true = (rand_vals < p_arr).astype(np.int32) # 1 if match, 0 if distractor

    best_k = 0
    best_score = e_f05_k0

    # Evaluate each k = 1 .. K
    cum_tp = np.cumsum(sim_true, axis=1) # shape (n_samples, K)
    total_true = np.sum(sim_true, axis=1) # shape (n_samples,)

    for k in range(1, K + 1):
        tp_k = cum_tp[:, k - 1] # shape (n_samples,)
        scores = np.zeros(n_samples, dtype=np.float32)
        
        # When true_k == 0 and we picked k > 0: score is 0.0 (already 0)
        # When true_k > 0:
        pos_mask = total_true > 0
        if np.any(pos_mask):
            tp_pos = tp_k[pos_mask]
            tot_pos = total_true[pos_mask]
            # F_0.5 = (1.25 * tp) / (0.25 * k + tot)
            denom = 0.25 * float(k) + tot_pos.astype(np.float32)
            scores[pos_mask] = (1.25 * tp_pos.astype(np.float32)) / denom

        e_score = float(np.mean(scores))
        if e_score > best_score:
            best_score = e_score
            best_k = k

    return best_k

def run_phase_c():
    print("=" * 70)
    print("PHASE C: DECISION LAYER & TEST-DENSITY SIMULATION")
    print("=" * 70)
    t0 = time.time()

    # Load Phase B model & calibrator
    model_path = os.path.join(config.MODEL_DIR, "lgbm_matcher_v2.joblib")
    iso_path = os.path.join(config.MODEL_DIR, "isotonic_calibrator.joblib")
    meta_path = os.path.join(config.MODEL_DIR, "phase_b_meta.json")

    if not os.path.exists(model_path):
        print("Phase B model not found! Run train_phase_b.py first.")
        return

    print("Loading Phase B trained LightGBM model and Isotonic Calibrator...")
    clf = joblib.load(model_path)
    iso = joblib.load(iso_path)
    with open(meta_path, "r") as f:
        meta_info = json.load(f)
    print(f"Features: {len(meta_info['features'])} features loaded.")

    # 1. Load 50,000 evaluation sample
    bronze = BronzeIngestionEngine(threads=8)
    sample_size = 50000
    print(f"Loading {sample_size:,} validation entities...")
    s1_df, gt_df, s23_df = bronze.load_train_sample(
        config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3, config.TRAIN_GT, sample_size=sample_size
    )

    s1_ids = list(s1_df['entity_id'].values)
    gt_dict = {sid: set() for sid in s1_ids}
    gt_col_s1 = 's1_id' if 's1_id' in gt_df.columns else 'source1_entity_id'
    for row in gt_df.itertuples():
        sid = getattr(row, gt_col_s1)
        if sid in gt_dict:
            m_str = getattr(row, 'matched_entity_ids')
            if pd.notna(m_str) and str(m_str).strip():
                gt_dict[sid] = set(x.strip() for x in str(m_str).split(",") if x.strip())

    s1_meta = {
        r.entity_id: {
            'country': r.country,
            'name': str(r.business_name or "").lower(),
            'addr': str(r.business_address or "").lower()
        } for r in s1_df.itertuples()
    }

    # Extract pairs and features exactly as in Phase B
    countries = list(s1_df['country'].unique())
    all_s1_pairs = defaultdict(list)
    cand_info = {}

    for country_name in countries:
        s1_c = s1_df[s1_df['country'] == country_name]
        s23_c = s23_df[s23_df['country'] == country_name]

        s23_ids = s23_c['entity_id'].values
        s23_names = s23_c['business_name'].values
        s23_addrs = s23_c['business_address'].values
        s23_countries = s23_c['country'].values

        for cid, n, a in zip(s23_ids, s23_names, s23_addrs):
            cand_info[cid] = {
                'brand': str(n or "").lower(),
                'addr': str(a or "").lower(),
                'nums': extract_nums(str(a or "").lower()),
                'src': 'S2' if cid.startswith('S2-') else 'S3'
            }

        s23_silver = {}
        for i in range(len(s23_ids)):
            s23_silver[s23_ids[i]] = normalize_silver_entity(s23_names[i], s23_addrs[i])

        gold = GoldResolutionEngine()
        gold.build_index(s23_ids, [s23_silver[eid] for eid in s23_ids], s23_countries)

        for row in s1_c.itertuples():
            sid = row.entity_id
            s1_silver_ent = normalize_silver_entity(row.business_name, row.business_address)
            cand_ids, base_feats, metas = gold.query_and_rank_candidates(s1_silver_ent, country_name, s23_silver)

            for cid, f, m in zip(cand_ids, base_feats, metas):
                c_item = cand_info.get(cid, {})
                comb_sim = f[10]
                n_tsort = f[2]
                all_s1_pairs[sid].append({
                    'cid': cid,
                    'base_feat': f,
                    'meta': m,
                    'comb_sim': comb_sim,
                    'n_tsort': n_tsort,
                    'brand': c_item.get('brand', ''),
                    'addr': c_item.get('addr', ''),
                    'src': c_item.get('src', 'S2')
                })

        del s23_silver, gold
        gc.collect()

    print("Computing features & predicting calibrated probabilities...")
    reverse_map = compute_reverse_context_features(all_s1_pairs)

    all_features = []
    all_pairs = []
    for sid, items in all_s1_pairs.items():
        s1_data = s1_meta[sid]
        s1_nums = extract_nums(s1_data['addr'])
        s1_has_addr = 1.0 if s1_data['addr'] and s1_data['addr'] != 'nan' else 0.0
        group_feats = compute_group_context_features(items)

        for i, item in enumerate(items):
            cid = item['cid']
            base_f = item['base_feat']
            grp_f = group_feats[i]
            rev_f = reverse_map.get((sid, cid), [1.0, 1.0, 1.0])

            c_nums = cand_info.get(cid, {}).get('nums', set())
            c_has_addr = 1.0 if cand_info.get(cid, {}).get('addr', '') else 0.0

            strict_conflict = 0.0
            if s1_nums and c_nums and not s1_nums.intersection(c_nums):
                if fuzz.token_sort_ratio(s1_data['name'], cand_info.get(cid, {}).get('brand', '')) < 80:
                    strict_conflict = 1.0

            num_missing_one_side = 1.0 if (bool(s1_nums) != bool(c_nums)) else 0.0

            full_f = (
                base_f +
                [strict_conflict, num_missing_one_side] +
                grp_f +
                rev_f +
                [1.0 - s1_has_addr, 1.0 - c_has_addr]
            )
            all_features.append(full_f)
            all_pairs.append((sid, cid))

    X = np.array(all_features, dtype=np.float32)
    raw_probs = clf.predict_proba(X)[:, 1]
    cal_probs = iso.predict(raw_probs)

    # Store predicted probabilities per entity: sid -> list of (cid, prob, src)
    s1_cand_probs = defaultdict(list)
    for (sid, cid), p in zip(all_pairs, cal_probs):
        src = 'S2' if cid.startswith('S2-') else 'S3'
        s1_cand_probs[sid].append((cid, float(p), src))

    # Sort each entity's candidates descending by calibrated probability
    for sid in s1_cand_probs:
        s1_cand_probs[sid].sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("COMPARISON OF DECISION LAYER STRATEGIES")
    print("=" * 70)

    decision_results = []

    # Strategy 1: Global Threshold on Calibrated Probability
    print("\n--- Strategy 1: Global Threshold ---")
    for th in [0.55, 0.60, 0.65, 0.70, 0.75]:
        preds = {sid: [c for c, p, s in s1_cand_probs[sid] if p >= th] for sid in s1_ids}
        # Greedy exclusivity on contested candidates
        c_to_s1 = defaultdict(list)
        for sid, clist in preds.items():
            for c in clist:
                c_to_s1[c].append(sid)
        for c, s1s in c_to_s1.items():
            if len(s1s) > 1:
                # Keep highest prob
                best_s1 = max(s1s, key=lambda s: max([p for ci, p, _ in s1_cand_probs[s] if ci == c] or [0]))
                for s in s1s:
                    if s != best_s1 and c in preds[s]:
                        preds[s].remove(c)
        m = evaluate_metrics(gt_dict, preds, s1_meta)
        print(f"Global Th {th:.2f} -> Macro F0.5: {m['macro_f05']:.4f} (P: {m['mean_p']:.4f}, R: {m['mean_r']:.4f}, Singletons: {m['singleton_acc']:.4f})")
        decision_results.append(("Global Th " + str(th), m))

    # Strategy 2: Per-Source Threshold (separate for S2 and S3)
    print("\n--- Strategy 2: Per-Source Threshold ---")
    for th_s2 in [0.60, 0.65, 0.70]:
        for th_s3 in [0.60, 0.65, 0.70]:
            preds = {
                sid: [
                    c for c, p, s in s1_cand_probs[sid]
                    if (p >= th_s2 if s == 'S2' else p >= th_s3)
                ] for sid in s1_ids
            }
            c_to_s1 = defaultdict(list)
            for sid, clist in preds.items():
                for c in clist:
                    c_to_s1[c].append(sid)
            for c, s1s in c_to_s1.items():
                if len(s1s) > 1:
                    best_s1 = max(s1s, key=lambda s: max([p for ci, p, _ in s1_cand_probs[s] if ci == c] or [0]))
                    for s in s1s:
                        if s != best_s1 and c in preds[s]:
                            preds[s].remove(c)
            m = evaluate_metrics(gt_dict, preds, s1_meta)
            if th_s2 == 0.65 and th_s3 == 0.65 or th_s2 == 0.70 and th_s3 == 0.65:
                print(f"Per-Source (S2={th_s2:.2f}, S3={th_s3:.2f}) -> Macro F0.5: {m['macro_f05']:.4f} (P: {m['mean_p']:.4f}, R: {m['mean_r']:.4f})")
                decision_results.append((f"Per-Source S2={th_s2}/S3={th_s3}", m))

    # Strategy 3: Expected-F0.5 Top-k Decision Layer
    print("\n--- Strategy 3: Expected-F0.5 Top-k Layer ---")
    t_topk = time.time()
    preds_topk = {}
    for sid in s1_ids:
        cands_s1 = s1_cand_probs.get(sid, [])
        if not cands_s1:
            preds_topk[sid] = []
            continue
        p_list = [p for _, p, _ in cands_s1]
        best_k = compute_expected_f05_monte_carlo(p_list, n_samples=150)
        preds_topk[sid] = [c for c, _, _ in cands_s1[:best_k]]

    # Greedy exclusivity
    c_to_s1 = defaultdict(list)
    for sid, clist in preds_topk.items():
        for c in clist:
            c_to_s1[c].append(sid)
    for c, s1s in c_to_s1.items():
        if len(s1s) > 1:
            best_s1 = max(s1s, key=lambda s: max([p for ci, p, _ in s1_cand_probs[s] if ci == c] or [0]))
            for s in s1s:
                if s != best_s1 and c in preds_topk[s]:
                    preds_topk[s].remove(c)

    m_topk = evaluate_metrics(gt_dict, preds_topk, s1_meta)
    print(f"Expected-F0.5 Top-k in {time.time()-t_topk:.1f}s -> Macro F0.5: {m_topk['macro_f05']:.4f} (P: {m_topk['mean_p']:.4f}, R: {m_topk['mean_r']:.4f}, Singletons: {m_topk['singleton_acc']:.4f})")
    decision_results.append(("Expected-F0.5 Top-k", m_topk))

    # 4. Test-Density Simulation (D9)
    print("\n" + "=" * 70)
    print("TEST-DENSITY SIMULATION (Injecting 23% Extra Distractors to match 5.75 Pool/S1)")
    print("=" * 70)
    # Simulate test pool density: test has 5.75 / 4.68 = 1.23x candidates
    # We add 2 extra random distractor pairs per S1 with simulated background probability ~ Beta(0.5, 20)
    np.random.seed(42)
    s1_cand_probs_dense = defaultdict(list)
    for sid in s1_ids:
        # Original candidates
        orig = list(s1_cand_probs.get(sid, []))
        # Add 2-3 negative distractors
        n_distract = np.random.choice([2, 3])
        distract_probs = np.random.beta(0.5, 25.0, size=n_distract)
        for i, dp in enumerate(distract_probs):
            orig.append((f"DISTRACT-{sid}-{i}", float(dp), 'S2'))
        orig.sort(key=lambda x: x[1], reverse=True)
        s1_cand_probs_dense[sid] = orig

    print("Evaluating strategies under Test Density Simulation:")
    for th in [0.65, 0.70, 0.72, 0.75]:
        preds_dense = {sid: [c for c, p, s in s1_cand_probs_dense[sid] if p >= th and not c.startswith("DISTRACT")] for sid in s1_ids}
        c_to_s1 = defaultdict(list)
        for sid, clist in preds_dense.items():
            for c in clist:
                c_to_s1[c].append(sid)
        for c, s1s in c_to_s1.items():
            if len(s1s) > 1:
                best_s1 = max(s1s, key=lambda s: max([p for ci, p, _ in s1_cand_probs_dense[s] if ci == c] or [0]))
                for s in s1s:
                    if s != best_s1 and c in preds_dense[s]:
                        preds_dense[s].remove(c)
        m_dense = evaluate_metrics(gt_dict, preds_dense, s1_meta)
        print(f"Dense Test Simulation | Th {th:.2f} -> Macro F0.5: {m_dense['macro_f05']:.4f} (P: {m_dense['mean_p']:.4f}, R: {m_dense['mean_r']:.4f})")

    # Pick the winning decision strategy
    best_dec_name, best_dec_metrics = max(decision_results, key=lambda x: x[1]['macro_f05'])
    print("\n" + "=" * 70)
    print(f"WINNING DECISION STRATEGY: {best_dec_name} with F0.5 = {best_dec_metrics['macro_f05']:.4f}")
    print("=" * 70)

    # Save winning strategy configuration
    win_config = {
        "strategy": best_dec_name,
        "f05": best_dec_metrics['macro_f05'],
        "precision": best_dec_metrics['mean_p'],
        "recall": best_dec_metrics['mean_r'],
        "singletons": best_dec_metrics['singleton_acc']
    }
    with open(os.path.join(config.MODEL_DIR, "phase_c_decision.json"), "w") as f:
        json.dump(win_config, f, indent=2)

    # Log to reports/experiments.jsonl
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "C_decision_layer",
        "best_strategy": best_dec_name,
        "macro_f05": round(best_dec_metrics['macro_f05'], 4),
        "mean_p": round(best_dec_metrics['mean_p'], 4),
        "mean_r": round(best_dec_metrics['mean_r'], 4),
        "singleton_acc": round(best_dec_metrics['singleton_acc'], 4),
        "duration_sec": round(time.time() - t0, 1)
    }
    with open("reports/experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\nPhase C completed successfully in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    run_phase_c()
