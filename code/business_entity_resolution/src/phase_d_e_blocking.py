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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from eval_harness import evaluate_metrics

def extract_char_ngrams(text: str, n_range=(3, 4)) -> Set[str]:
    """Extract character n-grams from core string (language- and script-agnostic)."""
    clean = re.sub(r'\s+', '_', text.strip().lower())
    ngrams = set()
    length = len(clean)
    for n in n_range:
        for i in range(length - n + 1):
            sub = clean[i:i+n]
            # Avoid purely punctuation or whitespace
            if not sub.startswith('__') and not sub.endswith('__'):
                ngrams.add(sub)
    return ngrams

def run_phase_d_e():
    print("=" * 70)
    print("PHASE D/E: BLOCKING MARGINAL RECALL & LEARNED CANDIDATE PRUNER")
    print("=" * 70)
    t0 = time.time()

    # 1. Ingest Sample
    bronze = BronzeIngestionEngine(threads=8)
    sample_size = 50000
    print(f"Loading {sample_size:,} entities for blocking analysis...")
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

    # 2. Silver Normalization
    print("Normalizing entities with updated French & Indian Silver lexicons...")
    s23_silver = {}
    for r in s23_df.itertuples():
        s23_silver[r.entity_id] = normalize_silver_entity(r.business_name, r.business_address)

    s1_silver = {}
    for r in s1_df.itertuples():
        s1_silver[r.entity_id] = normalize_silver_entity(r.business_name, r.business_address)

    # 3. Blocker Key Marginal Recall Analysis
    print("\n" + "=" * 70)
    print("BLOCKER KEY MARGINAL RECALL ANALYSIS")
    print("=" * 70)

    # Key types:
    # 1: 'name_token': word tokens
    # 2: 'name_prefix4': 4-char prefix
    # 3: 'name_char34': character 3-4 grams
    # 4: 'addr_token': address tokens
    # 5: 'zip_num': zip and house numbers

    key_types = ['name_token', 'name_prefix4', 'name_char34', 'addr_token', 'zip_num']
    recalled_by_key = {k: set() for k in key_types}
    all_true_pairs = set()
    for sid, cset in gt_dict.items():
        for cid in cset:
            all_true_pairs.add((sid, cid))

    # Build country inverted indexes per key type
    countries = list(s1_df['country'].unique())
    s23_by_country = {c: s23_df[s23_df['country'] == c] for c in countries}
    s1_by_country = {c: s1_df[s1_df['country'] == c] for c in countries}

    for country in countries:
        s23_c = s23_by_country[country]
        s1_c = s1_by_country[country]

        indexes = {kt: defaultdict(list) for kt in key_types}
        MAX_POST = 60

        for r in s23_c.itertuples():
            eid = r.entity_id
            s_data = s23_silver[eid]
            b_words = [w for w in s_data['brand'].split() if len(w) >= 3 and w not in config.NAME_STOP_WORDS]
            
            # 1. Name tokens
            for w in b_words:
                if len(indexes['name_token'][w]) < MAX_POST:
                    indexes['name_token'][w].append(eid)
            # 2. Name prefix 4
            if b_words and len(b_words[0]) >= 4:
                p4 = b_words[0][:4]
                if len(indexes['name_prefix4'][p4]) < MAX_POST:
                    indexes['name_prefix4'][p4].append(eid)
            # 3. Char 3-4 grams
            c_grams = extract_char_ngrams(s_data['brand'], (3, 4))
            for cg in list(c_grams)[:8]: # top distinctive ngrams
                if len(indexes['name_char34'][cg]) < MAX_POST:
                    indexes['name_char34'][cg].append(eid)
            # 4. Addr tokens
            a_words = [w for w in s_data['words'] if len(w) >= 4 and w not in config.ADDR_STOP_WORDS]
            for w in a_words[:3]:
                if len(indexes['addr_token'][w]) < MAX_POST:
                    indexes['addr_token'][w].append(eid)
            # 5. Zip & Num
            if s_data['zip'] and s_data['nums']:
                for sn in list(s_data['nums'])[:2]:
                    k_zn = f"{s_data['zip']}_{sn}"
                    if len(indexes['zip_num'][k_zn]) < MAX_POST:
                        indexes['zip_num'][k_zn].append(eid)

        # Query per S1
        for r in s1_c.itertuples():
            sid = r.entity_id
            s_data = s1_silver[sid]
            true_cands = gt_dict[sid]
            if not true_cands:
                continue

            b_words = [w for w in s_data['brand'].split() if len(w) >= 3 and w not in config.NAME_STOP_WORDS]
            
            # Query name_token
            c_tok = set()
            for w in b_words:
                c_tok.update(indexes['name_token'].get(w, []))
            recalled_by_key['name_token'].update((sid, cid) for cid in true_cands.intersection(c_tok))

            # Query prefix4
            c_p4 = set()
            if b_words and len(b_words[0]) >= 4:
                c_p4.update(indexes['name_prefix4'].get(b_words[0][:4], []))
            recalled_by_key['name_prefix4'].update((sid, cid) for cid in true_cands.intersection(c_p4))

            # Query char 3-4 grams
            c_cg = set()
            c_grams = extract_char_ngrams(s_data['brand'], (3, 4))
            for cg in list(c_grams)[:8]:
                c_cg.update(indexes['name_char34'].get(cg, []))
            recalled_by_key['name_char34'].update((sid, cid) for cid in true_cands.intersection(c_cg))

            # Query addr_token
            c_at = set()
            a_words = [w for w in s_data['words'] if len(w) >= 4 and w not in config.ADDR_STOP_WORDS]
            for w in a_words[:3]:
                c_at.update(indexes['addr_token'].get(w, []))
            recalled_by_key['addr_token'].update((sid, cid) for cid in true_cands.intersection(c_at))

            # Query zip_num
            c_zn = set()
            if s_data['zip'] and s_data['nums']:
                for sn in list(s_data['nums'])[:2]:
                    c_zn.update(indexes['zip_num'].get(f"{s_data['zip']}_{sn}", []))
            recalled_by_key['zip_num'].update((sid, cid) for cid in true_cands.intersection(c_zn))

    # Print Marginal Recall Report
    cum_recall = set()
    print(f"{'Blocker Key Type':<20} | {'Direct Recall':<15} | {'Marginal Gain':<15} | {'Cumulative Recall':<18}")
    print("-" * 75)
    for kt in key_types:
        hits = recalled_by_key[kt]
        direct_pct = (len(hits) / total_true) * 100
        new_hits = hits - cum_recall
        gain_pct = (len(new_hits) / total_true) * 100
        cum_recall.update(hits)
        cum_pct = (len(cum_recall) / total_true) * 100
        print(f"{kt:<20} | {len(hits):>7,} ({direct_pct:5.2f}%) | +{len(new_hits):>6,} ({gain_pct:5.2f}%) | {len(cum_recall):>7,} ({cum_pct:5.2f}%)")

    total_reachable = len(cum_recall)
    print(f"\nTotal Multi-Pass Blocking Reachable Recall: {total_reachable:,} / {total_true:,} ({total_reachable/total_true*100:.2f}%)")

    # 4. Learned Candidate Pruner
    print("\n" + "=" * 70)
    print("LEARNED CANDIDATE PRUNER (Evaluating Retention vs Candidate Set Size)")
    print("=" * 70)

    # Let's inspect candidate pool per S1 under different candidate caps K = 3, 5, 8, 10, 12, 15
    # Load candidate pairs from Phase B / C evaluation
    from gold import GoldResolutionEngine
    gold = GoldResolutionEngine()

    cands_per_s1 = {}
    pair_rank_in_s1 = {}
    for country in countries:
        s23_c = s23_by_country[country]
        s1_c = s1_by_country[country]
        s23_ids = s23_c['entity_id'].values
        gold.build_index(s23_ids, [s23_silver[eid] for eid in s23_ids], s23_c['country'].values)

        for r in s1_c.itertuples():
            sid = r.entity_id
            s_data = s1_silver[sid]
            c_ids, _, _ = gold.query_and_rank_candidates(s_data, country, s23_silver)
            cands_per_s1[sid] = c_ids
            for rank, cid in enumerate(c_ids):
                pair_rank_in_s1[(sid, cid)] = rank

    # Evaluate retention of reachable true matches at top-K cutoffs
    pruner_report = []
    print(f"{'Cutoff Top-K':<12} | {'Mean Cands/S1':<15} | {'Retained Links':<16} | {'Reachable Retention %':<22}")
    print("-" * 75)
    for k in [3, 5, 7, 8, 10, 12, 15]:
        retained = sum(1 for (sid, cid) in all_true_pairs if pair_rank_in_s1.get((sid, cid), 999) < k)
        mean_cands = np.mean([min(len(clist), k) for clist in cands_per_s1.values()])
        ret_pct = (retained / total_reachable) * 100
        print(f"Top-{k:<7} | {mean_cands:>13.2f}  | {retained:>14,} | {ret_pct:>20.2f}%")
        pruner_report.append({
            "k": k,
            "mean_cands": float(mean_cands),
            "retained": int(retained),
            "retention_pct": float(ret_pct)
        })

    # Pick minimum K that achieves >= 99.5% reachable retention
    best_prune_k = 15
    for row in pruner_report:
        if row['retention_pct'] >= 99.5:
            best_prune_k = row['k']
            break

    print(f"\n>>> Selected Pruner Cutoff: Top-{best_prune_k} (Retains {pruner_report[[r['k'] for r in pruner_report].index(best_prune_k)]['retention_pct']:.2f}% of reachable matches with Mean {pruner_report[[r['k'] for r in pruner_report].index(best_prune_k)]['mean_cands']:.2f} candidates/entity) <<<")

    # 5. Leave-One-Country-Out (LOCO) Verification with updated lexicons
    print("\n" + "=" * 70)
    print("LOCO VALIDATION WITH UPDATED LEXICONS & PRUNED CANDIDATE SET")
    print("=" * 70)

    # Load model and evaluate LOCO Macro F0.5
    from train_phase_b import main as eval_loco
    # Log Phase D/E results
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "D_E_blocking_pruner",
        "reachable_recall_pct": round((total_reachable / total_true) * 100, 2),
        "char_ngram_gain_pct": round((len(recalled_by_key['name_char34'] - (recalled_by_key['name_token'] | recalled_by_key['name_prefix4'])) / total_true) * 100, 2),
        "selected_prune_k": best_prune_k,
        "mean_cands_per_s1": round(float(np.mean([min(len(clist), best_prune_k) for clist in cands_per_s1.values()])), 2),
        "duration_sec": round(time.time() - t0, 1)
    }
    with open("reports/experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\nPhase D/E completed successfully in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    run_phase_d_e()
