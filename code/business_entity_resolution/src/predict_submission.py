"""
Production Test Inference Pipeline with Gated Phase B/C/D/E Components:
- 43-Feature Trained LightGBM Matcher
- Isotonic Probability Calibration
- Winning Asymmetric Decision Layer (S2=0.70, S3=0.65)
- Reverse & Group Context Features
- Global Greedy Bipartite Exclusivity
- Top-12 Pruned Candidate Generation
- Disk-Backed Batch Streaming for Complete Memory Safety
- Full Validation & Submission Packaging
"""

import os
import sys
import gc
import re
import time
import json
import shutil
import zipfile
import subprocess
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
from features_v2 import compute_group_context_features, extract_nums


def main():
    print("=" * 70, flush=True)
    print("STARTING PRODUCTION TEST INFERENCE (PHASE B/C/D/E GATED)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    # 1. Load trained models
    model_path = os.path.join(config.MODEL_DIR, "lgbm_matcher_v2.joblib")
    iso_path = os.path.join(config.MODEL_DIR, "isotonic_calibrator.joblib")
    meta_path = os.path.join(config.MODEL_DIR, "phase_b_meta.json")

    print("Loading models and calibrators...", flush=True)
    clf = joblib.load(model_path)
    iso = joblib.load(iso_path)
    with open(meta_path, "r") as f:
        meta_info = json.load(f)
    print(f"Loaded LightGBM matcher ({len(meta_info['features'])} features) and Isotonic Calibrator.", flush=True)

    # Decision thresholds from Phase C winner
    TH_S2 = 0.70
    TH_S3 = 0.65
    PRUNE_TOP_K = 12  # Retains 99.4% reachable recall while reducing candidate pool size
    print(f"Active Thresholds: S2 = {TH_S2:.2f}, S3 = {TH_S3:.2f} | Candidate Prune K = {PRUNE_TOP_K}", flush=True)

    bronze = BronzeIngestionEngine(threads=8)
    test_countries = bronze.get_test_countries(config.TEST_S1)
    all_s1_ordered = bronze.get_all_s1_ordered(config.TEST_S1)

    print(f"\nTotal Test S1 Entities: {len(all_s1_ordered):,}", flush=True)
    print("Countries:", flush=True)
    for c_name, c_cnt in test_countries:
        print(f"  - {c_name}: {c_cnt:,} entities", flush=True)

    temp_dir = os.path.join(config.OUTPUT_DIR, "temp_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Global tracking of candidate-to-S1 raw matches for global bipartite exclusivity
    cand_to_s1_matches = defaultdict(list)
    raw_s1_matches = defaultdict(list)

    # Process country by country
    for country_name, _ in test_countries:
        print(f"\n" + "=" * 60, flush=True)
        print(f"PROCESSING COUNTRY: {country_name}", flush=True)
        print("=" * 60, flush=True)
        t_country = time.time()

        # Ingest country data
        s1_c, s23_c = bronze.load_country_data(config.TEST_S1, config.TEST_S2, config.TEST_S3, country_name)
        print(f"Loaded: S1={len(s1_c):,}, Candidates={len(s23_c):,}", flush=True)

        # 1. Normalize candidates to Silver standard (single-pass, zero redundancy)
        print("Silver normalizing candidate pool...", flush=True)
        t_sil = time.time()
        s23_ids = s23_c['entity_id'].values
        s23_names = s23_c['business_name'].values
        s23_addrs = s23_c['business_address'].values
        s23_countries = s23_c['country'].values
        del s23_c
        gc.collect()

        s23_silver: Dict[str, dict] = {}
        for i in range(len(s23_ids)):
            s23_silver[s23_ids[i]] = normalize_silver_entity(s23_names[i], s23_addrs[i])
            if (i + 1) % 1000000 == 0:
                print(f"  Normalized {i+1:,} / {len(s23_ids):,} candidates ({time.time()-t_sil:.1f}s)...", flush=True)

        print(f"Silver normalized {len(s23_silver):,} candidates in {time.time()-t_sil:.1f}s.", flush=True)

        # 2. Build Gold inverted index (Top-12 prune)
        print("Building Gold inverted index...", flush=True)
        t_idx = time.time()
        gold = GoldResolutionEngine(top_k=PRUNE_TOP_K)
        gold.build_index(s23_ids, [s23_silver[eid] for eid in s23_ids], s23_countries)
        del s23_ids, s23_names, s23_addrs, s23_countries
        gc.collect()
        print(f"Gold index built in {time.time()-t_idx:.1f}s (Total keys: {len(gold.index):,}).", flush=True)

        # 3. Query blocking candidates in disk-backed batches
        print("Querying blocking candidates in disk-backed streaming batches...", flush=True)
        t_q = time.time()
        cand_to_s1s = defaultdict(list)
        BATCH_SIZE = 50000
        n_s1 = len(s1_c)
        num_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE

        country_cand_file = os.path.join(temp_dir, f"candidate_{country_name}.tsv")
        f_cand = open(country_cand_file, "w", encoding="utf-8")

        for b_idx in range(num_batches):
            b_start = b_idx * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1)
            batch_slice = s1_c.iloc[b_start:b_end]

            batch_queries = []
            for row in batch_slice.itertuples():
                sid = row.entity_id
                s1_silver_ent = normalize_silver_entity(row.business_name, row.business_address)
                cand_ids, base_feats, _ = gold.query_and_rank_candidates(s1_silver_ent, country_name, s23_silver)

                # Write candidate pairs chunk to disk
                f_cand.write(f"{sid}\t{','.join(cand_ids)}\n")

                # Track for reverse context
                for cid, f in zip(cand_ids, base_feats):
                    cand_to_s1s[cid].append((sid, f[10]))  # f[10] = comb_sim

                batch_queries.append((
                    sid,
                    str(row.business_name or "").lower(),
                    str(row.business_address or "").lower(),
                    cand_ids,
                    base_feats
                ))

            # Persist batch query results to temporary disk storage to keep RAM flat
            chunk_path = os.path.join(temp_dir, f"query_{country_name}_{b_idx}.joblib")
            joblib.dump(batch_queries, chunk_path, compress=1)
            del batch_queries, batch_slice
            gc.collect()

            print(f"  Queried {b_end:,} / {n_s1:,} S1 entities ({time.time()-t_q:.1f}s)...", flush=True)

        f_cand.close()
        print(f"Finished candidate querying for {country_name} in {time.time()-t_q:.1f}s.", flush=True)

        # Release gold index to free memory
        del gold
        gc.collect()

        # 4. Compute reverse context only for contested candidates (len > 1)
        print("Computing reverse context features on contested candidate pool...", flush=True)
        t_rev = time.time()
        contested_rev = {}
        for cid, s1_list in cand_to_s1s.items():
            if len(s1_list) > 1:
                s1_list.sort(key=lambda x: x[1], reverse=True)
                best_sid, best_sc = s1_list[0]
                second_sc = s1_list[1][1]
                cnt = float(len(s1_list))
                for rank, (sid, sc) in enumerate(s1_list):
                    if rank == 0:
                        contested_rev[(sid, cid)] = [cnt, 1.0, float(best_sc - second_sc)]
                    else:
                        contested_rev[(sid, cid)] = [cnt, 0.0, float(sc - best_sc)]

        del cand_to_s1s
        gc.collect()
        print(f"Reverse context built for {len(contested_rev):,} contested pairs in {time.time()-t_rev:.1f}s.", flush=True)

        # 5. Score candidate pairs batch-by-batch
        print("Extracting 43 dense features and scoring with calibrated LightGBM...", flush=True)
        t_score = time.time()
        for b_idx in range(num_batches):
            chunk_path = os.path.join(temp_dir, f"query_{country_name}_{b_idx}.joblib")
            batch_queries = joblib.load(chunk_path)
            os.remove(chunk_path)

            batch_features = []
            batch_pairs = []

            for sid, s1_name, s1_addr, cand_ids, base_feats in batch_queries:
                s1_nums = extract_nums(s1_addr)
                s1_has_addr = 1.0 if s1_addr and s1_addr != 'nan' else 0.0

                cand_items = []
                for cid, f in zip(cand_ids, base_feats):
                    c_data = s23_silver.get(cid, {})
                    cand_items.append({
                        'cid': cid,
                        'comb_sim': f[10],
                        'n_tsort': f[2],
                        'brand': c_data.get('brand', ''),
                        'src': 'S2' if cid.startswith('S2-') else 'S3'
                    })

                grp_feats = compute_group_context_features(cand_items)

                for i, item in enumerate(cand_items):
                    cid = item['cid']
                    base_f = base_feats[i]
                    gf = grp_feats[i]
                    rev_f = contested_rev.get((sid, cid), [1.0, 1.0, 1.0])

                    c_data = s23_silver.get(cid, {})
                    c_nums = c_data.get('nums', set())
                    c_has_addr = 1.0 if c_data.get('has_addr', False) else 0.0

                    strict_conflict = 0.0
                    if s1_nums and c_nums and not s1_nums.intersection(c_nums):
                        if fuzz.token_sort_ratio(s1_name, c_data.get('brand', '')) < 80:
                            strict_conflict = 1.0

                    num_missing_one_side = 1.0 if (bool(s1_nums) != bool(c_nums)) else 0.0

                    full_f = (
                        base_f +
                        [strict_conflict, num_missing_one_side] +
                        gf +
                        rev_f +
                        [1.0 - s1_has_addr, 1.0 - c_has_addr]
                    )
                    batch_features.append(full_f)
                    batch_pairs.append((sid, cid, item['src']))

            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                raw_probs = clf.predict_proba(X_batch)[:, 1]
                cal_probs = iso.predict(raw_probs)

                for (sid, cid, src), p in zip(batch_pairs, cal_probs):
                    th = TH_S2 if src == 'S2' else TH_S3
                    if p >= th:
                        prob_val = float(p)
                        cand_to_s1_matches[cid].append((sid, prob_val))
                        raw_s1_matches[sid].append((cid, prob_val))

            del batch_queries, batch_features, batch_pairs
            gc.collect()

            if (b_idx + 1) % 4 == 0 or (b_idx + 1) == num_batches:
                print(f"  Scored {min((b_idx+1)*BATCH_SIZE, n_s1):,} / {n_s1:,} S1 entities ({time.time()-t_score:.1f}s)...", flush=True)

        del s23_silver, contested_rev, s1_c
        gc.collect()
        print(f"Completed {country_name} in {time.time()-t_country:.1f}s.", flush=True)

    # 6. Global Greedy Bipartite Exclusivity
    print("\n" + "=" * 60, flush=True)
    print("APPLYING GLOBAL GREEDY BIPARTITE EXCLUSIVITY", flush=True)
    print("=" * 60, flush=True)
    t_bip = time.time()

    contested_count = sum(1 for cid, s1s in cand_to_s1_matches.items() if len(s1s) > 1)
    print(f"Total Contested Candidate IDs: {contested_count:,}", flush=True)

    cand_winner = {}
    for cid, s1_list in cand_to_s1_matches.items():
        if len(s1_list) > 1:
            best_s1 = max(s1_list, key=lambda x: x[1])[0]
            cand_winner[cid] = best_s1

    final_matches = {}
    total_links = 0
    singletons = 0

    for sid in all_s1_ordered:
        m_items = raw_s1_matches.get(sid, [])
        filtered = []
        for cid, prob in m_items:
            if cid in cand_winner and cand_winner[cid] != sid:
                continue
            filtered.append(cid)

        unique_matches = list(dict.fromkeys(filtered))
        final_matches[sid] = unique_matches
        total_links += len(unique_matches)
        if len(unique_matches) == 0:
            singletons += 1

    del cand_to_s1_matches, raw_s1_matches, cand_winner
    gc.collect()

    print(f"Bipartite exclusivity applied in {time.time()-t_bip:.1f}s.", flush=True)
    print(f"Final links: {total_links:,} (Average matches/entity: {total_links/len(all_s1_ordered):.2f})", flush=True)
    print(f"Final singletons: {singletons:,} ({singletons/len(all_s1_ordered)*100:.2f}%)", flush=True)

    # 7. Write Final Production Files in Exact Test Order
    print("\n" + "=" * 60, flush=True)
    print("WRITING FINAL OUTPUT FILES IN EXACT TEST SET ORDER", flush=True)
    print("=" * 60, flush=True)

    match_file = config.MATCHING_OUTPUT
    cand_file = config.CANDIDATE_OUTPUT

    print(f"Consolidating candidate pairs into {cand_file}...", flush=True)
    # Read chunk files into an ordered dictionary or stream directly
    cand_chunks_data = {}
    for c_name, _ in test_countries:
        chunk_c_file = os.path.join(temp_dir, f"candidate_{c_name}.tsv")
        if os.path.exists(chunk_c_file):
            with open(chunk_c_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        cand_chunks_data[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        cand_chunks_data[parts[0]] = ""

    with open(cand_file, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ordered:
            c_str = cand_chunks_data.get(sid, "")
            f_c.write(f"{sid}\t{c_str}\n")

    del cand_chunks_data
    gc.collect()

    print(f"Writing matching results to {match_file}...", flush=True)
    with open(match_file, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ordered:
            m_str = ",".join(final_matches.get(sid, []))
            f_m.write(f"{sid}\t{m_str}\n")

    del final_matches
    gc.collect()

    # Clean temporary chunk folder
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # 8. Run Official Validation Script
    print("\n" + "=" * 60, flush=True)
    print("RUNNING OFFICIAL VALIDATION SCRIPT", flush=True)
    print("=" * 60, flush=True)
    val_cmd = [
        sys.executable,
        os.path.join(config.BASE_DIR, "datasource", "utils", "validate_submission.py"),
        "--matching", match_file,
        "--candidate", cand_file,
        "--test-dir", config.TEST_DIR
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print("STDERR:", res.stderr, flush=True)

    # 9. Package Submission ZIP
    print("=" * 60, flush=True)
    print("PACKAGING SUBMISSION ZIP", flush=True)
    print("=" * 60, flush=True)
    zip_path = os.path.join(config.BASE_DIR, "Antigravity_submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(match_file, arcname="output/matching_results.tsv")
        zipf.write(cand_file, arcname="output/candidate_pairs.tsv")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Successfully packaged {zip_path} ({zip_size_mb:.2f} MB).", flush=True)
    print(f"\nFull execution completed successfully in {time.time()-t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
