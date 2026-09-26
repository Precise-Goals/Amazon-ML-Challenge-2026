"""
End-to-End Execution Pipeline for Business Entity Resolution.
Supports:
  --mode train     : Train model on train dataset & tune F_0.5 threshold
  --mode predict   : Run candidate generation & inference on test dataset
  --mode all       : Train model, optimize threshold, and generate test outputs
"""

import os
import sys
import time
import argparse
import duckdb
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils import (
    normalize_text,
    get_address_components,
    calculate_macro_f05
)
from blocking import BlockingEngine
from features import extract_pair_features
from model import EntityResolutionModel


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def train_pipeline(sample_size: int = 100000) -> EntityResolutionModel:
    """Train matching classifier and optimize threshold using representative training sample."""
    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)
    t0 = time.time()
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    
    print(f"Loading {sample_size:,} sample entities from train_source1.tsv...")
    s1_df = con.query(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TRAIN_S1}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        USING SAMPLE {sample_size} ROWS
    """).df()
    
    s1_ids = set(s1_df['entity_id'])
    
    print("Loading corresponding Ground Truth matches...")
    gt_df = con.query(f"""
        SELECT source1_entity_id as s1_id, matched_entity_ids
        FROM read_csv('{config.TRAIN_GT}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        WHERE source1_entity_id IN (SELECT entity_id FROM s1_df)
    """).df()
    
    gt_dict = {}
    true_matched_ids = set()
    for row in gt_df.itertuples():
        s1_id = row.s1_id
        m_str = row.matched_entity_ids
        if pd.notna(m_str) and str(m_str).strip():
            m_set = set(x.strip() for x in str(m_str).split(",") if x.strip())
            gt_dict[s1_id] = m_set
            true_matched_ids.update(m_set)
        else:
            gt_dict[s1_id] = set()
            
    # Ensure every sampled S1 entity has an entry
    for sid in s1_ids:
        if sid not in gt_dict:
            gt_dict[sid] = set()
            
    print(f"Sample contains {len(s1_df):,} S1 entities, {len(true_matched_ids):,} true match targets.")
    
    # Load candidate pool from S2 and S3 (all true matches + background noise sample)
    print("Loading candidate pool from S2 and S3...")
    s2_df = con.query(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TRAIN_S2}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        WHERE entity_id IN (SELECT unnest(string_split(matched_entity_ids, ',')) FROM read_csv('{config.TRAIN_GT}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE source1_entity_id IN (SELECT entity_id FROM s1_df))
        OR entity_id IN (SELECT entity_id FROM read_csv('{config.TRAIN_S2}', delim='\\t', header=True, auto_detect=True, all_varchar=True) USING SAMPLE 250000 ROWS)
    """).df()
    
    s3_df = con.query(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{config.TRAIN_S3}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        WHERE entity_id IN (SELECT unnest(string_split(matched_entity_ids, ',')) FROM read_csv('{config.TRAIN_GT}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE source1_entity_id IN (SELECT entity_id FROM s1_df))
        OR entity_id IN (SELECT entity_id FROM read_csv('{config.TRAIN_S3}', delim='\\t', header=True, auto_detect=True, all_varchar=True) USING SAMPLE 250000 ROWS)
    """).df()
    
    s23_pool = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Candidate pool size: {len(s23_pool):,} records.")
    
    print("Building Multi-Pass Blocking Inverted Index...")
    engine = BlockingEngine(max_postings=config.MAX_POSTINGS_PER_KEY)
    c_ids = s23_pool['entity_id'].values
    c_names = s23_pool['business_name'].values
    c_addrs = s23_pool['business_address'].values
    c_countries = s23_pool['country'].values
    engine.build_index_from_arrays(c_ids, c_names, c_addrs, c_countries)
    
    print("Pre-parsing candidate metadata for fast ranking...")
    cand_cache = {}
    for i in range(len(c_ids)):
        cid = c_ids[i]
        c_n = normalize_text(c_names[i], clean_domains=True)
        c_a = normalize_text(c_addrs[i], clean_domains=False)
        c_nums, c_zip, _, _ = get_address_components(c_a, config.ADDR_STOP_WORDS)
        cand_cache[cid] = (c_n, c_a, set(c_nums), c_zip)
        
    del s2_df, s3_df, s23_pool, c_ids, c_names, c_addrs, c_countries
    
    print("Querying and ranking candidates...")
    features = []
    labels = []
    pair_keys = []
    
    s1_all_ids = s1_df['entity_id'].values
    s1_all_names = s1_df['business_name'].values
    s1_all_addrs = s1_df['business_address'].values
    s1_all_countries = s1_df['country'].values
    
    for i in range(len(s1_all_ids)):
        s1_id = s1_all_ids[i]
        cands = engine.query(s1_id, s1_all_names[i], s1_all_addrs[i], s1_all_countries[i])
        true_set = gt_dict.get(s1_id, set())
        
        s1_name = normalize_text(s1_all_names[i], clean_domains=True)
        s1_addr = normalize_text(s1_all_addrs[i], clean_domains=False)
        s1_nums, s1_zip, _, _ = get_address_components(s1_addr, config.ADDR_STOP_WORDS)
        s1_num_set = set(s1_nums)
        
        # Dual combined candidate ranking: preserves transliterated & aliased entities
        scored = []
        for cid in cands:
            c_data = cand_cache.get(cid)
            if not c_data:
                continue
            c_n, c_a, c_num_set, c_zip = c_data
            
            n_sim = fuzz.token_set_ratio(s1_name, c_n)
            a_sim = fuzz.token_set_ratio(s1_addr, c_a) if s1_addr and c_a else 0.0
            num_m = bool(s1_num_set.intersection(c_num_set))
            zip_m = bool(s1_zip and c_zip and s1_zip == c_zip)
            
            c_score = (
                max(n_sim, a_sim, int(0.6 * n_sim + 0.4 * a_sim))
                + (20 if n_sim >= 50 and a_sim >= 40 else 0)
                + (10 if zip_m or num_m else 0)
            )
            scored.append((c_score, cid, c_data, n_sim, a_sim))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored[:config.TOP_K_CANDIDATES]
        
        for c_score, cid, c_data, n_sim, a_sim in top_candidates:
            c_n, c_a, c_num_set, c_zip = c_data
            feat = extract_pair_features(
                s1_name, s1_addr, s1_num_set, s1_zip,
                cid, c_n, c_a, c_num_set, c_zip,
                precomputed_name_tset=n_sim / 100.0,
                precomputed_addr_tset=a_sim / 100.0
            )
            features.append(feat)
            labels.append(1 if cid in true_set else 0)
            pair_keys.append((s1_id, cid))
            
    print(f"Generated {len(features):,} candidate pair feature vectors (Positives: {sum(labels):,}).")
    
    # Train / Validation Split (80% / 20%)
    cutoff = int(0.80 * len(s1_df))
    train_ids = set(s1_all_ids[:cutoff])
    val_ids = set(s1_all_ids[cutoff:])
    
    train_mask = [p[0] in train_ids for p in pair_keys]
    val_mask = [p[0] in val_ids for p in pair_keys]
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pair_keys = [pair_keys[i] for i, m in enumerate(val_mask) if m]
    
    print(f"Fitting LightGBM on {len(X_train):,} training pairs...")
    model_wrapper = EntityResolutionModel()
    model_wrapper.train(X_train, y_train)
    
    print("Optimizing classification threshold for Macro F_0.5...")
    best_th, best_f05 = model_wrapper.optimize_threshold(
        X_val, val_pair_keys, val_ids, gt_dict
    )
    
    print(f">>> Optimal Threshold: {best_th:.2f} | Validation Macro F_0.5: {best_f05:.4f} <<<")
    
    model_save_path = os.path.join(config.MODEL_DIR, "lgbm_model.joblib")
    model_wrapper.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    print(f"Training pipeline finished in {time.time()-t0:.2f}s.\n")
    return model_wrapper

def predict_pipeline(model_wrapper: EntityResolutionModel = None):
    """Run candidate generation and inference on the test set."""
    print("=" * 60)
    print("STARTING TEST INFERENCE PIPELINE")
    print("=" * 60)
    t0 = time.time()
    
    if model_wrapper is None:
        model_save_path = os.path.join(config.MODEL_DIR, "lgbm_model.joblib")
        if not os.path.exists(model_save_path):
            print("Trained model not found. Training model first...")
            model_wrapper = train_pipeline()
        else:
            print(f"Loading trained model from {model_save_path}...")
            model_wrapper = EntityResolutionModel.load(model_save_path)
            
    print(f"Active Threshold for F_0.5: {model_wrapper.threshold:.2f}")
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    
    # Get distinct countries in test_source1.tsv (supports US, India, France, and any open country!)
    print(f"Scanning test set from: {config.TEST_DIR}")
    countries_df = con.query(f"""
        SELECT country, count(*) as count 
        FROM read_csv('{config.TEST_S1}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        GROUP BY country ORDER BY count DESC
    """).df()
    
    print("Test Countries detected:")
    for _, r in countries_df.iterrows():
        print(f"  - {r['country']}: {int(r['count']):,} entities")
        
    all_s1_ordered = con.query(f"""
        SELECT entity_id FROM read_csv('{config.TEST_S1}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
    """).df()['entity_id'].tolist()
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(config.OUTPUT_DIR, "temp_chunks")
    # Clean temp_chunks to ensure fresh prediction
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process country-by-country to maintain low memory footprint and high parallel throughput
    for _, r in countries_df.iterrows():
        country_name = r['country']
        country_match_file = os.path.join(temp_dir, f"matching_{country_name}.tsv")
        country_cand_file = os.path.join(temp_dir, f"candidate_{country_name}.tsv")
        
        print(f"\n--- Processing Country: {country_name} ---", flush=True)
        t_country = time.time()
        
        # Load S1 for this country
        s1_c = con.query(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{config.TEST_S1}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            WHERE country = '{country_name}'
        """).df()
        
        # Load S2 and S3 for this country
        s23_c = con.query(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv('{config.TEST_S2}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE country = '{country_name}'
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv('{config.TEST_S3}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE country = '{country_name}'
        """).df()
        
        print(f"Loaded: S1={len(s1_c):,}, S23 Candidates={len(s23_c):,}", flush=True)
        
        ids = s23_c['entity_id'].values
        names = s23_c['business_name'].values
        addrs = s23_c['business_address'].values
        countries = s23_c['country'].values
        
        print("Building inverted index for country...", flush=True)
        engine = BlockingEngine(max_postings=config.MAX_POSTINGS_PER_KEY)
        engine.build_index_from_arrays(ids, names, addrs, countries)
        
        s23_lookup = dict(zip(ids, zip(names, addrs)))
        del s23_c, ids, names, addrs, countries
        
        s23_cache = {}
        def get_cand_data(cid):
            if cid in s23_cache:
                return s23_cache[cid]
            raw_data = s23_lookup.get(cid)
            if not raw_data:
                return None
            c_name = normalize_text(raw_data[0], clean_domains=True)
            c_addr = normalize_text(raw_data[1], clean_domains=False)
            c_nums, c_zip, _, _ = get_address_components(c_addr, config.ADDR_STOP_WORDS)
            data = (c_name, c_addr, set(c_nums), c_zip)
            s23_cache[cid] = data
            return data
            
        # Stream batch predictions directly to country files
        print("Querying blocking candidates and scoring...", flush=True)
        BATCH_SIZE = 50000
        
        with open(country_match_file, "w", encoding="utf-8") as f_match, \
             open(country_cand_file, "w", encoding="utf-8") as f_cand:
            
            for b_start in range(0, len(s1_c), BATCH_SIZE):
                batch_df = s1_c.iloc[b_start:b_start + BATCH_SIZE]
                
                batch_features = []
                batch_pairs = []
                batch_cands_map = {}
                
                for row in batch_df.itertuples():
                    s1_id = row.entity_id
                    cands = engine.query(s1_id, row.business_name, row.business_address, row.country)
                    
                    if not cands:
                        batch_cands_map[s1_id] = []
                        continue
                        
                    s1_name = normalize_text(row.business_name, clean_domains=True)
                    s1_addr = normalize_text(row.business_address, clean_domains=False)
                    s1_nums, s1_zip, _, _ = get_address_components(s1_addr, config.ADDR_STOP_WORDS)
                    s1_num_set = set(s1_nums)
                    
                    # Pre-rank to top candidates
                    scored = []
                    for cid in cands:
                        c_data = get_cand_data(cid)
                        if not c_data:
                            continue
                        c_name, c_addr, c_num_set, c_zip = c_data
                        
                        n_sim = fuzz.token_set_ratio(s1_name, c_name)
                        a_sim = fuzz.token_set_ratio(s1_addr, c_addr) if s1_addr and c_addr else 0.0
                        num_m = bool(s1_num_set.intersection(c_num_set))
                        zip_m = bool(s1_zip and c_zip and s1_zip == c_zip)
                        
                        c_score = (
                            max(n_sim, a_sim, int(0.6 * n_sim + 0.4 * a_sim))
                            + (20 if n_sim >= 50 and a_sim >= 40 else 0)
                            + (10 if zip_m or num_m else 0)
                        )
                        scored.append((c_score, cid, c_data, n_sim, a_sim))
                        
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_candidates = scored[:config.TOP_K_CANDIDATES]
                    
                    cand_ids = [x[1] for x in top_candidates]
                    batch_cands_map[s1_id] = cand_ids
                    
                    for c_score, cid, c_data, n_sim, a_sim in top_candidates:
                        c_name, c_addr, c_num_set, c_zip = c_data
                        feat = extract_pair_features(
                            s1_name, s1_addr, s1_num_set, s1_zip,
                            cid, c_name, c_addr, c_num_set, c_zip,
                            precomputed_name_tset=n_sim / 100.0,
                            precomputed_addr_tset=a_sim / 100.0
                        )
                        batch_features.append(feat)
                        batch_pairs.append((s1_id, cid))
                        
                # Score batch
                batch_matches_map = {row.entity_id: [] for row in batch_df.itertuples()}
                if batch_features:
                    X_batch = np.array(batch_features, dtype=np.float32)
                    probs = model_wrapper.predict_proba(X_batch)
                    
                    for (s1_id, cid), prob in zip(batch_pairs, probs):
                        if prob >= model_wrapper.threshold:
                            batch_matches_map[s1_id].append(cid)
                            
                # Write batch rows immediately to disk
                for row in batch_df.itertuples():
                    s1_id = row.entity_id
                    c_list = batch_cands_map.get(s1_id, [])
                    m_list = [m for m in dict.fromkeys(batch_matches_map.get(s1_id, [])) if m in c_list]
                    
                    f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")
                    f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                    
                print(f"  Processed {min(b_start + BATCH_SIZE, len(s1_c)):,} / {len(s1_c):,} S1 entities in {country_name}...", flush=True)
                
        # Clear country lookups from RAM
        del s23_lookup, s23_cache, s1_c, engine
        import gc
        gc.collect()
        print(f"Finished {country_name} in {time.time()-t_country:.2f}s.", flush=True)
        
    print("\n" + "=" * 60)
    print("ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST ORDER")
    print("=" * 60, flush=True)
    
    # Build fast dictionary of results from chunk files
    print("Reading chunk files into final files...", flush=True)
    all_chunks_match = {}
    all_chunks_cand = {}
    
    for _, r in countries_df.iterrows():
        c_name = r['country']
        with open(os.path.join(temp_dir, f"matching_{c_name}.tsv"), "r", encoding="utf-8") as f:
            for line in f:
                s1, _, rest = line.partition("\t")
                all_chunks_match[s1] = rest.rstrip("\n")
                
        with open(os.path.join(temp_dir, f"candidate_{c_name}.tsv"), "r", encoding="utf-8") as f:
            for line in f:
                s1, _, rest = line.partition("\t")
                all_chunks_cand[s1] = rest.rstrip("\n")
                
    # 1. Write matching_results.tsv
    print(f"Saving {config.MATCHING_OUTPUT}...", flush=True)
    matched_count = 0
    singleton_count = 0
    with open(config.MATCHING_OUTPUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ordered:
            val = all_chunks_match.get(s1_id, "")
            if val:
                matched_count += 1
            else:
                singleton_count += 1
            f.write(f"{s1_id}\t{val}\n")
            
    # 2. Write candidate_pairs.tsv
    print(f"Saving {config.CANDIDATE_OUTPUT}...", flush=True)
    total_cands_written = 0
    with open(config.CANDIDATE_OUTPUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ordered:
            val = all_chunks_cand.get(s1_id, "")
            if val:
                total_cands_written += len(val.split(","))
            f.write(f"{s1_id}\t{val}\n")
            
    avg_cands = total_cands_written / float(len(all_s1_ordered))
    print(f"\nInference Summary:")
    print(f"  Total S1 Entities Processed: {len(all_s1_ordered):,}")
    print(f"  Entities with Matches: {matched_count:,} ({matched_count/len(all_s1_ordered)*100:.2f}%)")
    print(f"  Singletons (Empty Matches): {singleton_count:,} ({singleton_count/len(all_s1_ordered)*100:.2f}%)")
    print(f"  Avg Candidates per S1: {avg_cands:.2f}")
    print(f"  Total Inference Time: {time.time()-t0:.2f}s")
    
    # 3. Automatic Validation Check
    validator_path = os.path.join(config.BASE_DIR, "datasource", "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        print("\n" + "=" * 60)
        print("RUNNING OFFICIAL VALIDATION SCRIPT")
        print("=" * 60)
        import subprocess
        val_cmd = [
            sys.executable,
            validator_path,
            "--matching", config.MATCHING_OUTPUT,
            "--candidate", config.CANDIDATE_OUTPUT,
            "--test-dir", config.TEST_DIR
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print(">>> OFFICIAL VALIDATION: PASSED SUCCESSFULLY! Output files are 100% compliant. <<<")
        else:
            print(f">>> OFFICIAL VALIDATION FAILED with exit code {res.returncode}. <<<")

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.TRAIN_SAMPLE_SIZE,
        help=f"Number of S1 train records to sample for model fitting (default: {config.TRAIN_SAMPLE_SIZE})"
    )
    args = parser.parse_args()
    
    if args.mode in ("train", "all"):
        model = train_pipeline(sample_size=args.sample_size)
    else:
        model = None
        
    if args.mode in ("predict", "all"):
        predict_pipeline(model_wrapper=model)

if __name__ == "__main__":
    main()
