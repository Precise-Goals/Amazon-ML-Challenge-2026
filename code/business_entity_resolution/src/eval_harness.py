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
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from gold import GoldResolutionEngine
from model import EntityResolutionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Local Evaluation Harness for Medallion Architecture")
    parser.add_argument("--sample-size", type=int, default=0, help="Number of S1 entities to evaluate (0 for full train)")
    parser.add_argument("--output-json", type=str, default="reports/eval_results.json", help="Path to save detailed JSON")
    return parser.parse_args()

def extract_nums(addr: str) -> Set[str]:
    raw = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in raw if len(n) <= 5}

def evaluate_metrics(gt_dict: Dict[str, Set[str]], preds_dict: Dict[str, List[str]], s1_meta: Dict[str, dict] = None):
    """Compute Macro F0.5, P, R, singleton accuracy, true-k breakdown, and country breakdown."""
    f05_list = []
    p_list = []
    r_list = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Bucket trackers
    k_buckets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], '6+': []}
    country_f05 = defaultdict(list)
    s2_f05 = []
    s3_f05 = []
    
    for sid, true_set in gt_dict.items():
        pred_set = set(preds_dict.get(sid, []))
        cntry = s1_meta[sid]['country'] if s1_meta and sid in s1_meta else 'Unknown'
        k = len(true_set)
        bucket_key = k if k <= 5 else '6+'
        
        if len(true_set) == 0 and len(pred_set) == 0:
            score = 1.0
            p = 1.0
            r = 1.0
        elif len(true_set) == 0 and len(pred_set) > 0:
            score = 0.0
            p = 0.0
            r = 1.0
            total_fp += len(pred_set)
        elif len(true_set) > 0 and len(pred_set) == 0:
            score = 0.0
            p = 0.0
            r = 0.0
            total_fn += len(true_set)
        else:
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            den = 0.25 * p + r
            score = (1.25 * p * r) / den if den > 0 else 0.0
            
        f05_list.append(score)
        p_list.append(p)
        r_list.append(r)
        k_buckets[bucket_key].append(score)
        country_f05[cntry].append(score)
        
        # Source-specific F0.5
        for src, target_list in [('S2', s2_f05), ('S3', s3_f05)]:
            t_src = {c for c in true_set if c.startswith(src)}
            p_src = {c for c in pred_set if c.startswith(src)}
            if len(t_src) == 0 and len(p_src) == 0:
                target_list.append(1.0)
            elif len(t_src) == 0 and len(p_src) > 0:
                target_list.append(0.0)
            elif len(t_src) > 0 and len(p_src) == 0:
                target_list.append(0.0)
            else:
                tp_s = len(t_src.intersection(p_src))
                fp_s = len(p_src - t_src)
                fn_s = len(t_src - p_src)
                p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
                r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
                den_s = 0.25 * p_s + r_s
                target_list.append((1.25 * p_s * r_s) / den_s if den_s > 0 else 0.0)

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    return {
        "macro_f05": float(np.mean(f05_list)),
        "mean_p": float(np.mean(p_list)),
        "mean_r": float(np.mean(r_list)),
        "micro_p": float(overall_p),
        "micro_r": float(overall_r),
        "singleton_acc": float(np.mean(k_buckets[0])) if k_buckets[0] else 0.0,
        "k_buckets": {str(k): float(np.mean(v)) if v else 0.0 for k, v in k_buckets.items()},
        "country_f05": {c: float(np.mean(v)) if v else 0.0 for c, v in country_f05.items()},
        "s2_f05": float(np.mean(s2_f05)) if s2_f05 else 0.0,
        "s3_f05": float(np.mean(s3_f05)) if s3_f05 else 0.0
    }

def main():
    args = parse_args()
    print("=" * 70)
    print("PHASE A2: LOCAL EVALUATION HARNESS (EXACT TEST PIPELINE ON TRAIN)")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Load trained model
    model_save_path = os.path.join(config.MODEL_DIR, "lgbm_model.joblib")
    if not os.path.exists(model_save_path):
        print("Trained LightGBM model not found. Training model first...")
        from pipeline import train_pipeline
        model_wrapper = train_pipeline(sample_size=100000)
    else:
        print(f"Loading trained LightGBM model from {model_save_path}...")
        model_wrapper = EntityResolutionModel.load(model_save_path)
        
    print(f"Model active threshold: {model_wrapper.threshold:.2f}")
    
    # 2. Ingest Train Data
    bronze = BronzeIngestionEngine(threads=8)
    sample_size = args.sample_size
    
    if sample_size > 0:
        print(f"\n--- Loading Stratified Sample: {sample_size:,} S1 entities ---")
        s1_df, gt_df, s23_df = bronze.load_train_sample(
            config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3, config.TRAIN_GT, sample_size=sample_size
        )
    else:
        print("\n--- Loading Full Train Dataset (2.2M S1, 10.3M Candidates) ---")
        # Load all train S1 and GT
        import duckdb
        con = duckdb.connect()
        s1_df = con.query(f"SELECT * FROM read_csv('{config.TRAIN_S1}', delim='\\t', header=true)").df()
        gt_df = con.query(f"SELECT * FROM read_csv('{config.TRAIN_GT}', delim='\\t', header=true)").df()
        # s23 loaded per country to save RAM
        s23_df = None
        del con
        gc.collect()

    s1_ids = set(s1_df['entity_id'])
    gt_dict = {sid: set() for sid in s1_ids}
    gt_col_s1 = 's1_id' if 's1_id' in gt_df.columns else 'source1_entity_id'
    for row in gt_df.itertuples():
        sid = getattr(row, gt_col_s1)
        if sid in gt_dict:
            m_str = getattr(row, 'matched_entity_ids')
            if pd.notna(m_str) and str(m_str).strip():
                gt_dict[sid] = set(x.strip() for x in str(m_str).split(",") if x.strip())
                
    total_true = sum(len(v) for v in gt_dict.values())
    total_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"Loaded {len(s1_df):,} S1 entities ({total_singletons:,} singletons = {total_singletons/len(s1_df)*100:.2f}%).")
    print(f"Total true links: {total_true:,}")

    # Build s1_meta for breakdown
    s1_meta = {
        r.entity_id: {
            'country': r.country,
            'name': str(r.business_name or "").lower(),
            'addr': str(r.business_address or "").lower()
        } for r in s1_df.itertuples()
    }
    
    # 3. Country-by-Country Processing (Exact Test Code Path)
    countries = list(s1_df['country'].unique())
    print(f"\nProcessing countries: {countries}")
    
    all_s1_cands: Dict[str, List[str]] = {}
    # candidate pair raw scores: sid -> list of dict(cid=cid, prob=prob, sc=sc, conflict=conflict)
    raw_scored_pairs: Dict[str, List[dict]] = defaultdict(list)
    cand_txt: Dict[str, Tuple[str, str]] = {}
    
    entity_full_recall = 0
    total_recalled_pairs = 0
    cand_counts = []
    
    for country_name in countries:
        print(f"\n--- Country: {country_name} ---")
        t_c = time.time()
        s1_c = s1_df[s1_df['country'] == country_name]
        
        if s23_df is not None:
            s23_c = s23_df[s23_df['country'] == country_name]
        else:
            s1_dummy, s23_c = bronze.load_country_data(config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3, country_name)
            del s1_dummy
            
        print(f"Loaded: S1={len(s1_c):,}, Candidates={len(s23_c):,}")
        
        # Silver normalization
        print("Silver normalization...")
        s23_ids = s23_c['entity_id'].values
        s23_names = s23_c['business_name'].values
        s23_addrs = s23_c['business_address'].values
        s23_countries = s23_c['country'].values
        
        for cid, n, a in zip(s23_ids, s23_names, s23_addrs):
            cand_txt[cid] = (str(n or "").lower(), str(a or "").lower())
            
        s23_silver = {}
        for i in range(len(s23_ids)):
            s23_silver[s23_ids[i]] = normalize_silver_entity(s23_names[i], s23_addrs[i])
            
        del s23_c
        gc.collect()
        
        # Gold Index
        print("Building Gold inverted index...")
        gold = GoldResolutionEngine()
        gold.build_index(s23_ids, [s23_silver[eid] for eid in s23_ids], s23_countries)
        del s23_ids, s23_names, s23_addrs, s23_countries
        gc.collect()
        
        # Batch Querying & Model Scoring
        print("Querying blocking candidates and scoring pairs...")
        BATCH_SIZE = 50000
        for b_start in range(0, len(s1_c), BATCH_SIZE):
            batch_df = s1_c.iloc[b_start:b_start + BATCH_SIZE]
            batch_features = []
            batch_pairs = []
            batch_metas = []
            
            for row in batch_df.itertuples():
                s1_id = row.entity_id
                s1_silver_ent = normalize_silver_entity(row.business_name, row.business_address)
                cand_ids, feats, metas = gold.query_and_rank_candidates(s1_silver_ent, country_name, s23_silver)
                
                all_s1_cands[s1_id] = cand_ids
                cand_counts.append(len(cand_ids))
                
                # Check recall
                t_set = gt_dict[s1_id]
                c_set = set(cand_ids)
                if t_set:
                    total_recalled_pairs += len(t_set.intersection(c_set))
                    if t_set.issubset(c_set):
                        entity_full_recall += 1
                else:
                    entity_full_recall += 1
                    
                for cid, f, m in zip(cand_ids, feats, metas):
                    batch_features.append(f)
                    batch_pairs.append((s1_id, cid))
                    batch_metas.append(m)
                    
            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = model_wrapper.predict_proba(X_batch)
                
                for (sid, cid), m, prob in zip(batch_pairs, batch_metas, probs):
                    s_name, s_addr = s1_meta[sid]['name'], s1_meta[sid]['addr']
                    c_name, c_addr = cand_txt.get(cid, ("", ""))
                    
                    n_sim = fuzz.token_set_ratio(s_name, c_name)
                    a_sim = fuzz.token_set_ratio(s_addr, c_addr) if s_addr and c_addr else 0.0
                    sc = 0.6 * n_sim + 0.4 * a_sim
                    
                    s_nums = extract_nums(s_addr)
                    c_nums = extract_nums(c_addr)
                    has_num_conflict = False
                    if s_nums and c_nums and not s_nums.intersection(c_nums):
                        tsort = fuzz.token_sort_ratio(s_name, c_name)
                        if tsort < 80:
                            has_num_conflict = True
                            
                    passed_gate = GoldResolutionEngine.apply_precision_gate(m, prob, model_wrapper.threshold)
                    
                    raw_scored_pairs[sid].append({
                        'cid': cid,
                        'prob': float(prob),
                        'sc': float(sc),
                        'num_conflict': has_num_conflict,
                        'passed_gate': bool(passed_gate)
                    })
                    
        del s23_silver, gold
        gc.collect()
        print(f"Finished {country_name} in {time.time()-t_c:.1f}s.")

    # 4. Blocking Report
    print("\n" + "=" * 70)
    print("BLOCKING RECALL & EFFICIENCY REPORT")
    print("=" * 70)
    ent_rec_ceiling = (entity_full_recall / len(s1_df)) * 100
    pair_rec = (total_recalled_pairs / total_true) * 100 if total_true > 0 else 100.0
    mean_cands = float(np.mean(cand_counts))
    median_cands = float(np.median(cand_counts))
    p95_cands = float(np.percentile(cand_counts, 95))
    
    print(f"Entity-level recall ceiling (100% of matches found): {ent_rec_ceiling:.2f}%")
    print(f"Pair-level recall (recalled true links / total true links): {pair_rec:.2f}% ({total_recalled_pairs:,} / {total_true:,})")
    print(f"Candidates per S1: Mean={mean_cands:.2f}, Median={median_cands:.1f}, P95={p95_cands:.1f}")

    # 5. Phase A3: 8 Ablations of {bipartite, number-veto, cap-3}
    print("\n" + "=" * 70)
    print("PHASE A3: 8 ABLATIONS OF {bipartite, number-veto, cap-3}")
    print("=" * 70)
    
    ablation_results = []
    
    combos = [
        (0, 0, 0, "1. Raw Model (No heuristics)"),
        (1, 0, 0, "2. +Bipartite Only"),
        (0, 1, 0, "3. +Number Veto Only"),
        (0, 0, 1, "4. +Cap-3 Only"),
        (1, 1, 0, "5. +Bipartite + Number Veto"),
        (1, 0, 1, "6. +Bipartite + Cap-3"),
        (0, 1, 1, "7. +Number Veto + Cap-3"),
        (1, 1, 1, "8. Full Consolidated (Bipartite + Veto + Cap-3)")
    ]
    
    for use_bip, use_veto, use_cap, label in combos:
        # Step 1: Initial filtering
        current_preds: Dict[str, List[dict]] = {}
        for sid, p_list in raw_scored_pairs.items():
            filtered = []
            for item in p_list:
                if not item['passed_gate']:
                    continue
                if use_veto and item['num_conflict']:
                    continue
                filtered.append(item)
            current_preds[sid] = filtered
            
        # Step 2: Bipartite mutual exclusivity
        if use_bip:
            cand_to_s1 = defaultdict(list)
            for sid, items in current_preds.items():
                for item in items:
                    cand_to_s1[item['cid']].append((sid, item['sc']))
                    
            cand_winner = {}
            for cid, s1_list in cand_to_s1.items():
                if len(s1_list) > 1:
                    best_s1 = max(s1_list, key=lambda x: x[1])[0]
                    cand_winner[cid] = best_s1
                    
            for sid in current_preds:
                current_preds[sid] = [
                    item for item in current_preds[sid] 
                    if item['cid'] not in cand_winner or cand_winner[item['cid']] == sid
                ]
                
        # Step 3: Cap-3 per source
        final_dict: Dict[str, List[str]] = {}
        for sid, items in current_preds.items():
            if use_cap:
                # Sort by score descending
                sorted_items = sorted(items, key=lambda x: x['sc'], reverse=True)
                s2_c = 0
                s3_c = 0
                res = []
                for item in sorted_items:
                    cid = item['cid']
                    if cid.startswith('S2-'):
                        if s2_c >= 3:
                            continue
                        s2_c += 1
                    elif cid.startswith('S3-'):
                        if s3_c >= 3:
                            continue
                        s3_c += 1
                    res.append(cid)
                final_dict[sid] = res
            else:
                final_dict[sid] = [item['cid'] for item in items]
                
        # Compute metrics
        m = evaluate_metrics(gt_dict, final_dict, s1_meta)
        ablation_results.append({
            "name": label,
            "bipartite": use_bip,
            "number_veto": use_veto,
            "cap_3": use_cap,
            "metrics": m
        })
        
        print(f"\n{label}:")
        print(f"  Macro F0.5: {m['macro_f05']:.4f} | Precision: {m['mean_p']:.4f} (micro: {m['micro_p']:.4f}) | Recall: {m['mean_r']:.4f} (micro: {m['micro_r']:.4f})")
        print(f"  Singleton Acc: {m['singleton_acc']:.4f} | S2 F0.5: {m['s2_f05']:.4f} | S3 F0.5: {m['s3_f05']:.4f}")
        print(f"  Country F0.5: US={m['country_f05'].get('US', 0):.4f}, India={m['country_f05'].get('India', 0):.4f}")
        print(f"  True-k: k=0: {m['k_buckets']['0']:.4f}, k=1: {m['k_buckets']['1']:.4f}, k=2: {m['k_buckets']['2']:.4f}, k=3: {m['k_buckets']['3']:.4f}, k=4: {m['k_buckets']['4']:.4f}, k=5: {m['k_buckets']['5']:.4f}, k=6+: {m['k_buckets']['6+']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE (Macro F0.5 Comparison)")
    print("=" * 70)
    summary_rows = []
    base_f05 = ablation_results[0]['metrics']['macro_f05']
    for r in ablation_results:
        f05 = r['metrics']['macro_f05']
        summary_rows.append({
            "Combo": r['name'],
            "F0.5": f05,
            "Delta_vs_Raw": f"{f05 - base_f05:+.4f}",
            "Prec": r['metrics']['mean_p'],
            "Rec": r['metrics']['mean_r'],
            "Singleton": r['metrics']['singleton_acc']
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # Log to reports/experiments.jsonl
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "A2_A3_local_eval_ablation",
        "sample_size": len(s1_df),
        "blocking_recall_ceiling": round(ent_rec_ceiling, 2),
        "pair_recall": round(pair_rec, 2),
        "mean_cands_per_s1": round(mean_cands, 2),
        "ablations": [
            {
                "name": r['name'],
                "bipartite": r['bipartite'],
                "number_veto": r['number_veto'],
                "cap_3": r['cap_3'],
                "macro_f05": round(r['metrics']['macro_f05'], 4),
                "mean_p": round(r['metrics']['mean_p'], 4),
                "mean_r": round(r['metrics']['mean_r'], 4),
                "singleton_acc": round(r['metrics']['singleton_acc'], 4),
                "country_f05": {k: round(v, 4) for k, v in r['metrics']['country_f05'].items()},
                "k_buckets": {k: round(v, 4) for k, v in r['metrics']['k_buckets'].items()}
            } for r in ablation_results
        ],
        "duration_sec": round(time.time() - t0, 1)
    }
    with open("reports/experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\nResults successfully logged to reports/experiments.jsonl in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
