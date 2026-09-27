"""
Pre-scores all test universe candidate pairs (p >= 0.30) into a single compact parquet cache.
Allows instant (5-second) threshold sweeps, per-country tuning, and submission generation.
"""

import os
import sys
import glob
import json
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from bronze import BronzeIngestionEngine
from universe import FEATURE_NAMES
from decision_v3 import exclusive_best_mask

CACHE_PATH = os.path.join(config.BASE_DIR, "artifacts", "test_universe", "test_scored_cache.parquet")
MODEL_DIR = os.path.join(config.MODEL_DIR, "v3")
WORK_DIR = os.path.join(config.BASE_DIR, "artifacts", "test_universe")
OUT_DIR = config.OUTPUT_DIR


def build_cache(min_p=0.30):
    t0 = time.time()
    models = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(MODEL_DIR, "lgbm_fold*.txt")))]
    iso = joblib.load(os.path.join(MODEL_DIR, "isotonic.joblib"))
    print(f"Loaded {len(models)} fold models and isotonic calibrator.")
    
    parquets = sorted(glob.glob(os.path.join(WORK_DIR, "feat_*.parquet")))
    print(f"Scoring {len(parquets)} parquet chunks (keeping p >= {min_p})...")
    
    keep = []
    n_total = 0
    for idx, p in enumerate(parquets, 1):
        t_chunk = time.time()
        d = pd.read_parquet(p)
        n_total += len(d)
        X = d[FEATURE_NAMES].to_numpy(np.float32)
        pr = np.mean([m.predict(X) for m in models], axis=0)
        pc = iso.predict(pr).astype(np.float32)
        m = pc >= min_p
        if np.any(m):
            keep.append(pd.DataFrame({
                "sid": d["sid"].to_numpy()[m],
                "cid": d["cid"].to_numpy()[m],
                "country": d["country"].to_numpy()[m],
                "is_s3": d["is_s3"].to_numpy(dtype=np.int8)[m],
                "p": pc[m],
            }))
        print(f"[{idx}/{len(parquets)}] {os.path.basename(p)}: {len(d):,} rows -> {np.sum(m):,} kept ({time.time()-t_chunk:.1f}s)")
    
    cached = pd.concat(keep, ignore_index=True)
    cached.to_parquet(CACHE_PATH, index=False)
    print(f"WROTE cache to {CACHE_PATH}: {len(cached):,} pairs retained from {n_total:,} total ({time.time()-t0:.1f}s, {os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")
    return cached


def load_or_build_cache():
    if os.path.exists(CACHE_PATH):
        print(f"Loading existing cache from {CACHE_PATH} ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)...")
        return pd.read_parquet(CACHE_PATH)
    return build_cache()


def evaluate_thresholds(df, th_india_s2, th_india_s3, th_us_s2, th_us_s3, th_fr_s2, th_fr_s3,
                        generate_tag="", s1_path=None):
    t0 = time.time()
    # Vectorized per-row threshold
    c = df["country"].to_numpy()
    s3 = df["is_s3"].to_numpy().astype(bool)
    
    th = np.zeros(len(df), dtype=np.float32)
    # India
    m_in = c == "India"
    th[m_in & ~s3] = th_india_s2
    th[m_in & s3] = th_india_s3
    # US
    m_us = c == "US"
    th[m_us & ~s3] = th_us_s2
    th[m_us & s3] = th_us_s3
    # France
    m_fr = c == "France"
    th[m_fr & ~s3] = th_fr_s2
    th[m_fr & s3] = th_fr_s3
    
    passed = df["p"].to_numpy() >= th
    kept = df[passed].copy()
    
    # Exclusivity
    codes, _ = pd.factorize(kept["cid"])
    kept = kept[exclusive_best_mask(kept["p"].to_numpy(), codes)]
    
    # Ground truth targets for comparison
    gt_targets = {
        "India": {"s1": 809986, "links_target": 3.4645, "empty_target": 0.0559},
        "US": {"s1": 663106, "links_target": 3.4591, "empty_target": 0.0558},
        "France": {"s1": 259452, "links_target": 3.46, "empty_target": 0.0560},
    }
    
    total_s1 = 1732544
    n_links = len(kept)
    matched_s1 = kept["sid"].nunique()
    empty_s1 = total_s1 - matched_s1
    
    print("\n" + "=" * 78)
    print(f"EVALUATION: India({th_india_s2:.3f}/{th_india_s3:.3f}), US({th_us_s2:.3f}/{th_us_s3:.3f}), FR({th_fr_s2:.3f}/{th_fr_s3:.3f})")
    print(f"Total Links: {n_links:,} | Mean Links/S1: {n_links/total_s1:.4f} | Empty Rate: {empty_s1/total_s1:.2%}")
    print("-" * 78)
    print(f"{'Country':<10} {'S1 Count':<10} {'Links':<10} {'Links/S1':<10} {'Empty %':<10} {'GT Links':<10} {'GT Empty %':<10}")
    print("-" * 78)
    for cname, tgt in gt_targets.items():
        sub = kept[kept["country"] == cname]
        c_links = len(sub)
        c_matched = sub["sid"].nunique()
        c_empty_rate = 1.0 - c_matched / tgt["s1"]
        print(f"{cname:<10} {tgt['s1']:<10,d} {c_links:<10,d} {c_links/tgt['s1']:<10.4f} {c_empty_rate:<10.2%} {tgt['links_target']:<10.4f} {tgt['empty_target']:<10.2%}")
    print("=" * 78)

    if generate_tag:
        if s1_path is None:
            s1_path = os.path.join(config.DATASET_DIR, "test", "test_source1.tsv")
        order = BronzeIngestionEngine().get_all_s1_ordered(s1_path)
        matches = kept.sort_values("p", ascending=False).groupby("sid")["cid"].apply(list).to_dict()
        
        cands = {}
        for cf in glob.glob(os.path.join(WORK_DIR, "cands_*.tsv")):
            with open(cf, encoding="utf-8") as f:
                for line in f:
                    sid, _, lst = line.rstrip("\n").partition("\t")
                    cands[sid] = lst
                    
        mpath = os.path.join(OUT_DIR, f"matching_results_{generate_tag}.tsv")
        cpath = os.path.join(OUT_DIR, f"candidate_pairs_{generate_tag}.tsv")
        with open(mpath, "w", encoding="utf-8", newline="\n") as fm, \
             open(cpath, "w", encoding="utf-8", newline="\n") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid in order:
                ml = list(dict.fromkeys(matches.get(sid, [])))
                cl = cands.get(sid, "")
                fm.write(f"{sid}\t{','.join(ml)}\n")
                fc.write(f"{sid}\t{cl}\n")
        print(f"\nGENERATED {mpath} ({os.path.getsize(mpath)/1e6:.1f} MB) in {time.time()-t0:.1f}s")
        
        val = os.path.join(config.BASE_DIR, "datasource", "utils", "validate_submission.py")
        te = os.path.join(config.DATASET_DIR, "test")
        if os.path.exists(val):
            r = subprocess.run([sys.executable, val, "--matching", mpath, "--candidate", cpath,
                                "--test-dir", te], capture_output=True, text=True)
            print("VALIDATOR RESULT:")
            print((r.stdout + r.stderr).strip()[-1000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true", help="Force rebuild test predictions cache")
    ap.add_argument("--tag", default="", help="Generate submission files with this tag")
    ap.add_argument("--th-india", type=float, default=0.55)
    ap.add_argument("--th-us", type=float, default=0.65)
    ap.add_argument("--th-fr", type=float, default=0.62)
    ap.add_argument("--s3-delta", type=float, default=0.02, help="Extra threshold added for S3")
    args = ap.parse_args()

    if args.build_cache:
        df = build_cache()
    else:
        df = load_or_build_cache()

    th_in_s2 = args.th_india
    th_in_s3 = args.th_india + args.s3_delta
    th_us_s2 = args.th_us
    th_us_s3 = args.th_us + args.s3_delta
    th_fr_s2 = args.th_fr
    th_fr_s3 = args.th_fr + args.s3_delta

    evaluate_thresholds(df, th_in_s2, th_in_s3, th_us_s2, th_us_s3, th_fr_s2, th_fr_s3,
                        generate_tag=args.tag)


if __name__ == "__main__":
    main()
