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
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from gold import GoldResolutionEngine
from features_v2 import compute_group_context_features, compute_reverse_context_features, extract_nums
from eval_harness import evaluate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Phase B: Trained Matcher with Context & LOCO Validation")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of S1 entities to evaluate")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of GroupKFold splits")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("PHASE B: TRAINED MATCHER WITH GROUP & REVERSE CONTEXT")
    print("=" * 70)
    t0 = time.time()

    # 1. Load Data
    bronze = BronzeIngestionEngine(threads=8)
    sample_size = args.sample_size
    print(f"Loading {sample_size:,} stratified training entities...")
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

    total_true = sum(len(v) for v in gt_dict.values())
    print(f"Loaded {len(s1_ids):,} entities with {total_true:,} true links.")

    s1_meta = {
        r.entity_id: {
            'country': r.country,
            'name': str(r.business_name or "").lower(),
            'addr': str(r.business_address or "").lower()
        } for r in s1_df.itertuples()
    }

    # 2. Silver Normalization & Indexing per country
    countries = list(s1_df['country'].unique())
    all_s1_pairs = defaultdict(list)
    cand_info = {}

    for country_name in countries:
        print(f"\nProcessing Country: {country_name}...")
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

    # 3. Compute Group Context & Reverse Context Features
    print("\nComputing Group Context and Reverse Context features...")
    t_feat = time.time()
    reverse_map = compute_reverse_context_features(all_s1_pairs)

    feature_names = [
        # Base 24 features
        'n_ratio', 'n_partial', 'n_tsort', 'n_tset', 'n_wratio',
        'a_ratio', 'a_partial', 'a_tsort', 'a_tset',
        'max_sim', 'comb_sim', 'dual_high',
        'num_overlap', 'num_match', 'num_conflict',
        'zip_match', 'zip_conflict',
        'addr_missing', 'legal_match', 'legal_conflict',
        'name_exact', 'brand_len', 'word_cnt', 'has_translit',
        # Former Vetoes / Detailed Numbers
        'strict_num_conflict', 'num_missing_one_side',
        # Group Context (12)
        'rank_comb', 'rank_name', 'gap_comb', 'gap_name', 'z_comb',
        'n_cands', 'n_s2', 'n_s3', 'rank_within_src',
        'agree_max_other', 'agree_mean_other', 'count_sim_gt_90',
        # Reverse Context (3)
        'cand_s1_count', 'is_mutual_best', 'gap_to_second_s1',
        # Missing Field Indicators (2)
        's1_addr_missing', 'c_addr_missing'
    ]

    all_features = []
    all_labels = []
    all_pairs = []
    all_groups = []

    for sid, items in all_s1_pairs.items():
        true_set = gt_dict[sid]
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

            # Strict number conflict feature
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

            label = 1 if cid in true_set else 0

            all_features.append(full_f)
            all_labels.append(label)
            all_pairs.append((sid, cid))
            all_groups.append(sid)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    groups = np.array(all_groups)

    print(f"Engineered {len(feature_names)} features across {len(X):,} candidate pairs in {time.time()-t_feat:.1f}s.")
    print(f"Positive pairs: {sum(y):,} | Negative pairs: {len(y)-sum(y):,}")

    # 4. 5-Fold GroupKFold Cross-Validation
    print(f"\n--- Running {args.n_folds}-Fold GroupKFold Cross-Validation ---")
    gkf = GroupKFold(n_splits=args.n_folds)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 4
    }

    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Training Fold {fold}/{args.n_folds} (Train: {len(train_idx):,}, Val: {len(val_idx):,})...")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr, y_tr)

        val_preds = clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_preds
        fold += 1

    # 5. Probability Calibration (Isotonic Regression)
    print("\nFitting Isotonic Calibration on OOF Probabilities...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_probs, y)
    oof_calibrated = iso.predict(oof_probs)

    # 6. Tune Threshold on OOF Macro F0.5
    print("\nOptimizing Threshold on Honest OOF Macro F0.5...")
    best_th = 0.50
    best_f05 = 0.0
    best_metrics = None

    pair_sid = [p[0] for p in all_pairs]
    pair_cid = [p[1] for p in all_pairs]

    threshold_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for th in threshold_grid:
        oof_preds = defaultdict(list)
        for sid, cid, p in zip(pair_sid, pair_cid, oof_calibrated):
            if p >= th:
                oof_preds[sid].append(cid)

        # Apply soft bipartite exclusivity on contested candidates
        cand_to_s1 = defaultdict(list)
        for sid, c_list in oof_preds.items():
            for c in c_list:
                cand_to_s1[c].append(sid)

        for c, s1s in cand_to_s1.items():
            if len(s1s) > 1:
                # Find best S1 for this candidate
                best_s1 = max(s1s, key=lambda s: max([p for (si, ci), p in zip(all_pairs, oof_calibrated) if si == s and ci == c] or [0]))
                for s in s1s:
                    if s != best_s1 and c in oof_preds[s]:
                        oof_preds[s].remove(c)

        m = evaluate_metrics(gt_dict, oof_preds, s1_meta)
        print(f"Threshold {th:.2f} -> OOF Macro F0.5: {m['macro_f05']:.4f} (P: {m['mean_p']:.4f}, R: {m['mean_r']:.4f}, Singletons: {m['singleton_acc']:.4f})")
        if m['macro_f05'] > best_f05:
            best_f05 = m['macro_f05']
            best_th = th
            best_metrics = m

    print(f"\n>>> Peak OOF Macro F0.5: {best_f05:.4f} at Optimal Threshold = {best_th:.2f} <<<")

    # 7. Leave-One-Country-Out (LOCO) Generalization Check (Proxy for France)
    print("\n" + "=" * 70)
    print("LEAVE-ONE-COUNTRY-OUT (LOCO) GENERALIZATION AUDIT")
    print("=" * 70)
    is_us = np.array([s1_meta[p[0]]['country'] == 'US' for p in all_pairs])
    is_in = ~is_us

    # Train on US -> Test on India
    clf_us = lgb.LGBMClassifier(**lgb_params)
    clf_us.fit(X[is_us], y[is_us])
    in_probs = clf_us.predict_proba(X[is_in])[:, 1]
    
    in_preds = defaultdict(list)
    for (sid, cid), p in zip(np.array(all_pairs)[is_in], in_probs):
        if p >= best_th:
            in_preds[sid].append(cid)
    gt_in = {sid: gt_dict[sid] for sid in gt_dict if s1_meta[sid]['country'] == 'India'}
    m_in = evaluate_metrics(gt_in, in_preds, s1_meta)
    print(f"Train on US -> Test on India: F0.5 = {m_in['macro_f05']:.4f} (P: {m_in['mean_p']:.4f}, R: {m_in['mean_r']:.4f})")

    # Train on India -> Test on US
    clf_in = lgb.LGBMClassifier(**lgb_params)
    clf_in.fit(X[is_in], y[is_in])
    us_probs = clf_in.predict_proba(X[is_us])[:, 1]

    us_preds = defaultdict(list)
    for (sid, cid), p in zip(np.array(all_pairs)[is_us], us_probs):
        if p >= best_th:
            us_preds[sid].append(cid)
    gt_us = {sid: gt_dict[sid] for sid in gt_dict if s1_meta[sid]['country'] == 'US'}
    m_us = evaluate_metrics(gt_us, us_preds, s1_meta)
    print(f"Train on India -> Test on US: F0.5 = {m_us['macro_f05']:.4f} (P: {m_us['mean_p']:.4f}, R: {m_us['mean_r']:.4f})")

    mean_loco_f05 = (m_in['macro_f05'] + m_us['macro_f05']) / 2.0
    print(f"Mean LOCO Macro F0.5: {mean_loco_f05:.4f}")

    # 8. Train Final Full Model on all pairs and save
    print("\nFitting final model on all data and saving artifacts...")
    final_clf = lgb.LGBMClassifier(**lgb_params)
    final_clf.fit(X, y)

    import joblib
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(final_clf, os.path.join(config.MODEL_DIR, "lgbm_matcher_v2.joblib"))
    joblib.dump(iso, os.path.join(config.MODEL_DIR, "isotonic_calibrator.joblib"))
    with open(os.path.join(config.MODEL_DIR, "phase_b_meta.json"), "w") as f:
        json.dump({"threshold": best_th, "features": feature_names}, f)

    # 9. Log to reports/experiments.jsonl
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "B_trained_matcher",
        "sample_size": sample_size,
        "n_features": len(feature_names),
        "oof_macro_f05": round(best_f05, 4),
        "oof_p": round(best_metrics['mean_p'], 4),
        "oof_r": round(best_metrics['mean_r'], 4),
        "singleton_acc": round(best_metrics['singleton_acc'], 4),
        "optimal_threshold": best_th,
        "loco_india_f05": round(m_in['macro_f05'], 4),
        "loco_us_f05": round(m_us['macro_f05'], 4),
        "mean_loco_f05": round(mean_loco_f05, 4),
        "duration_sec": round(time.time() - t0, 1)
    }
    with open("reports/experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\nPhase B successfully completed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
