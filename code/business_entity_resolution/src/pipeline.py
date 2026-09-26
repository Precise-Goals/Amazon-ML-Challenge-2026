"""
Medallion Architecture Execution Pipeline for Business Entity Resolution.
Supports:
  --mode train     : Train model on train dataset & tune F_0.5 threshold
  --mode predict   : Run candidate generation & inference on test dataset
  --mode all       : Train model, optimize threshold, and generate test outputs
"""

import os
import sys
import gc
import time
import argparse
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from gold import GoldResolutionEngine
from model import EntityResolutionModel
from utils import calculate_macro_f05


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def train_pipeline(sample_size: int = 100000) -> EntityResolutionModel:
    """Train matching classifier and optimize threshold using representative training sample."""
    print("=" * 60)
    print("STARTING MEDALLION TRAINING PIPELINE")
    print("=" * 60)
    t0 = time.time()
    
    # 1. Bronze Layer Ingestion
    print(f"Bronze Layer: Loading {sample_size:,} sample entities from train...")
    bronze = BronzeIngestionEngine(threads=8)
    s1_df, gt_df, s23_df = bronze.load_train_sample(
        config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3, config.TRAIN_GT, sample_size=sample_size
    )
    
    s1_ids = set(s1_df['entity_id'])
    gt_dict: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
    for row in gt_df.itertuples():
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            gt_dict[row.s1_id] = set(x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip())
            
    total_true = sum(len(v) for v in gt_dict.values())
    singletons_true = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"Bronze loaded: {len(s1_df):,} S1 entities, {len(s23_df):,} candidates.")
    print(f"Ground Truth: {total_true:,} true matches ({singletons_true:,} singletons = {singletons_true/len(s1_df)*100:.2f}%).")
    
    # 2. Silver Layer Normalization
    print("Silver Layer: Standardizing text, extracting legal forms & transliterating...")
    t_silver = time.time()
    s23_silver = {}
    for row in s23_df.itertuples():
        s23_silver[row.entity_id] = normalize_silver_entity(row.business_name, row.business_address)
        
    s1_silver = {}
    for row in s1_df.itertuples():
        s1_silver[row.entity_id] = normalize_silver_entity(row.business_name, row.business_address)
    print(f"Silver normalized {len(s23_silver)+len(s1_silver):,} records in {time.time()-t_silver:.2f}s.")
    
    # 3. Gold Layer Inverted Indexing
    print("Gold Layer: Building multi-pass high-recall inverted index...")
    t_idx = time.time()
    gold = GoldResolutionEngine()
    gold.build_index(
        s23_df['entity_id'].values,
        [s23_silver[eid] for eid in s23_df['entity_id'].values],
        s23_df['country'].values
    )
    print(f"Gold index built in {time.time()-t_idx:.2f}s (Total keys: {len(gold.index):,}).")
    
    # 4. Gold Layer Querying & Feature Generation
    print("Gold Layer: Querying candidates and generating 24 dense features...")
    t_feat = time.time()
    features = []
    labels = []
    pair_keys = []
    pair_meta = []
    
    s1_list = s1_df['entity_id'].values
    s1_cntry = s1_df['country'].values
    recalled_cand = 0
    
    for i in range(len(s1_list)):
        sid = s1_list[i]
        cntry = s1_cntry[i]
        s1_data = s1_silver[sid]
        true_set = gt_dict[sid]
        
        cand_ids, feats, metas = gold.query_and_rank_candidates(s1_data, cntry, s23_silver)
        if true_set:
            recalled_cand += len(true_set.intersection(cand_ids))
            
        for cid, f, m in zip(cand_ids, feats, metas):
            features.append(f)
            labels.append(1 if cid in true_set else 0)
            pair_keys.append((sid, cid))
            pair_meta.append(m)
            
    print(f"Candidate Blocking Recall: {recalled_cand:,} / {total_true:,} ({recalled_cand/total_true*100:.2f}%)")
    print(f"Generated {len(features):,} candidate pairs in {time.time()-t_feat:.2f}s (Positives: {sum(labels):,}).")
    
    # 5. Train / Validation Split (80% / 20%)
    cutoff = int(0.80 * len(s1_list))
    train_sids = set(s1_list[:cutoff])
    val_sids = set(s1_list[cutoff:])
    
    train_mask = [p[0] in train_sids for p in pair_keys]
    val_mask = [p[0] in val_sids for p in pair_keys]
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pair_keys = [pair_keys[i] for i, m in enumerate(val_mask) if m]
    val_metas = [pair_meta[i] for i, m in enumerate(val_mask) if m]
    
    print(f"Fitting LightGBM on {len(X_train):,} training pairs...")
    model_wrapper = EntityResolutionModel()
    model_wrapper.train(X_train, y_train)
    
    print("Optimizing classification threshold for Macro F_0.5 with Gold Quality Gates...")
    best_th, best_f05 = model_wrapper.optimize_threshold(
        X_val, val_pair_keys, val_sids, gt_dict, val_metas=val_metas
    )
    
    print(f">>> Optimal Threshold: {best_th:.2f} | Validation Macro F_0.5: {best_f05:.4f} <<<")
    
    model_save_path = os.path.join(config.MODEL_DIR, "lgbm_model.joblib")
    model_wrapper.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    print(f"Training pipeline finished in {time.time()-t0:.2f}s.\n")
    return model_wrapper

def predict_pipeline(model_wrapper: EntityResolutionModel = None):
    """Run candidate generation and inference on the test set partitioned by country."""
    print("=" * 60)
    print("STARTING TEST INFERENCE PIPELINE (MEDALLION GOLD ARCHITECTURE)")
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
            
    print(f"Active Decision Threshold for F_0.5: {model_wrapper.threshold:.2f}")
    
    bronze = BronzeIngestionEngine(threads=8)
    test_countries = bronze.get_test_countries(config.TEST_S1)
    all_s1_ordered = bronze.get_all_s1_ordered(config.TEST_S1)
    
    print("Test Countries detected:")
    for c_name, c_cnt in test_countries:
        print(f"  - {c_name}: {c_cnt:,} entities")
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(config.OUTPUT_DIR, "temp_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process country-by-country to ensure peak speed and minimal memory footprint
    for country_name, _ in test_countries:
        country_match_file = os.path.join(temp_dir, f"matching_{country_name}.tsv")
        country_cand_file = os.path.join(temp_dir, f"candidate_{country_name}.tsv")
        
        print(f"\n--- Processing Country Partition: {country_name} ---", flush=True)
        t_country = time.time()
        
        # 1. Bronze Load
        s1_c, s23_c = bronze.load_country_data(config.TEST_S1, config.TEST_S2, config.TEST_S3, country_name)
        print(f"Loaded: S1={len(s1_c):,}, S23 Candidates={len(s23_c):,}", flush=True)
        
        # 2. Silver Normalization for S23 candidate pool
        print("Normalizing candidate pool to Silver standard...", flush=True)
        t_silver = time.time()
        s23_ids = s23_c['entity_id'].values
        s23_names = s23_c['business_name'].values
        s23_addrs = s23_c['business_address'].values
        s23_countries = s23_c['country'].values
        del s23_c
        
        s23_silver: Dict[str, dict] = {}
        for i in range(len(s23_ids)):
            s23_silver[s23_ids[i]] = normalize_silver_entity(s23_names[i], s23_addrs[i])
        print(f"Normalized {len(s23_silver):,} candidates in {time.time()-t_silver:.2f}s.", flush=True)
        
        # 3. Gold Inverted Indexing
        print("Building Gold multi-pass inverted index...", flush=True)
        gold = GoldResolutionEngine()
        gold.build_index(s23_ids, [s23_silver[eid] for eid in s23_ids], s23_countries)
        del s23_ids, s23_names, s23_addrs, s23_countries
        gc.collect()
        
        # 4. Stream Batch Predictions
        print("Querying blocking candidates and scoring with Gold Quality Gates...", flush=True)
        BATCH_SIZE = 50000
        
        with open(country_match_file, "w", encoding="utf-8") as f_match, \
             open(country_cand_file, "w", encoding="utf-8") as f_cand:
            
            for b_start in range(0, len(s1_c), BATCH_SIZE):
                batch_df = s1_c.iloc[b_start:b_start + BATCH_SIZE]
                
                batch_features = []
                batch_pairs = []
                batch_metas = []
                batch_cands_map = {}
                
                for row in batch_df.itertuples():
                    s1_id = row.entity_id
                    s1_silver_ent = normalize_silver_entity(row.business_name, row.business_address)
                    
                    cand_ids, feats, metas = gold.query_and_rank_candidates(
                        s1_silver_ent, country_name, s23_silver
                    )
                    batch_cands_map[s1_id] = cand_ids
                    
                    for cid, feat, meta in zip(cand_ids, feats, metas):
                        batch_features.append(feat)
                        batch_pairs.append((s1_id, cid))
                        batch_metas.append(meta)
                        
                # Score batch
                batch_matches_map = {row.entity_id: [] for row in batch_df.itertuples()}
                if batch_features:
                    X_batch = np.array(batch_features, dtype=np.float32)
                    probs = model_wrapper.predict_proba(X_batch)
                    
                    for (s1_id, cid), meta, prob in zip(batch_pairs, batch_metas, probs):
                        if GoldResolutionEngine.apply_precision_gate(meta, prob, model_wrapper.threshold):
                            batch_matches_map[s1_id].append(cid)
                            
                # Write batch rows immediately to disk
                for row in batch_df.itertuples():
                    s1_id = row.entity_id
                    c_list = batch_cands_map.get(s1_id, [])
                    m_list = [m for m in dict.fromkeys(batch_matches_map.get(s1_id, [])) if m in c_list]
                    
                    f_cand.write(f"{s1_id}\t{','.join(c_list)}\n")
                    f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                    
                print(f"  Processed {min(b_start + BATCH_SIZE, len(s1_c)):,} / {len(s1_c):,} S1 entities in {country_name}...", flush=True)
                
        # Clean country lookups from RAM
        del s23_silver, s1_c, gold
        gc.collect()
        print(f"Finished {country_name} in {time.time()-t_country:.2f}s.", flush=True)
        
    print("\n" + "=" * 60)
    print("ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST ORDER")
    print("=" * 60, flush=True)
    
    # Read chunk files into memory
    print("Reading chunk files...", flush=True)
    all_chunks_match = {}
    all_chunks_cand = {}
    
    for c_name, _ in test_countries:
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
    total_matches_predicted = 0
    
    with open(config.MATCHING_OUTPUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ordered:
            val = all_chunks_match.get(s1_id, "")
            if val:
                matched_count += 1
                total_matches_predicted += len(val.split(","))
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
    avg_matches = total_matches_predicted / float(matched_count) if matched_count > 0 else 0
    print(f"\nInference Summary:")
    print(f"  Total S1 Entities Processed: {len(all_s1_ordered):,}")
    print(f"  Entities with Matches: {matched_count:,} ({matched_count/len(all_s1_ordered)*100:.2f}%)")
    print(f"  Singletons (Empty Matches): {singleton_count:,} ({singleton_count/len(all_s1_ordered)*100:.2f}%)")
    print(f"  Avg Matches per Non-Empty Entity: {avg_matches:.2f}")
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
