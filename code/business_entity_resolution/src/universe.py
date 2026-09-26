"""
Universe featurizer: the single code path that turns (S1, S2, S3) TSVs into scored-ready
candidate-pair feature chunks. Used identically by training (on train files) and inference
(on test files), so train/test feature skew is impossible by construction.

Why this exists (v2 defects it removes):
  * v2 trained on a 50k-S1 sample whose pool held ALL true matches but only ~5% of the other
    records, so neighbourhood competition (chains, co-tenants, other entities' duplicates)
    and reverse-context features (cand_s1_count, is_mutual_best) were ~20x weaker than on test.
  * v2 computed several features differently in train vs test (raw vs silver brand for
    agreement features, raw vs silver street numbers, NaN addresses counted as present in
    train, top_k 15 in train vs 12 in test).
Here every S1 of a country is queried against that country's FULL pool, and features come
from one function, whatever the split.
"""

import os
import gc
import time
import zlib
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional

from config import config
from bronze import BronzeIngestionEngine
from silver import normalize_silver_entity
from gold import GoldResolutionEngine
from features_v2 import compute_group_context_features

BASE_FEATURES = [
    'n_ratio', 'n_partial', 'n_tsort', 'n_tset', 'n_wratio',
    'a_ratio', 'a_partial', 'a_tsort', 'a_tset',
    'max_sim', 'comb_sim', 'dual_high',
    'num_overlap', 'num_match', 'num_conflict',
    'zip_match', 'zip_conflict',
    'addr_missing', 'legal_match', 'legal_conflict',
    'name_exact', 'brand_len', 'word_cnt', 'has_translit',
]
FEATURE_NAMES = BASE_FEATURES + [
    'strict_num_conflict', 'num_missing_one_side',
    'rank_comb', 'rank_name', 'gap_comb', 'gap_name', 'z_comb',
    'n_cands', 'n_s2', 'n_s3', 'rank_within_src',
    'agree_max_other', 'agree_mean_other', 'count_sim_gt_90',
    'cand_s1_count', 'is_mutual_best', 'gap_to_second_s1',
    's1_addr_missing', 'c_addr_missing',
]
UNCONTESTED_REV = [1.0, 1.0, 1.0]


def s1_bucket(sid: str) -> int:
    """Deterministic 0..999 bucket per S1 id (fold = bucket % 5, sampling = bucket // 5)."""
    return zlib.crc32(sid.encode("utf-8")) % 1000


def _reverse_context(cand_to_s1s: Dict[str, list]) -> Dict[tuple, List[float]]:
    rev = {}
    for cid, s1_list in cand_to_s1s.items():
        if len(s1_list) < 2:
            continue
        s1_list.sort(key=lambda x: x[1], reverse=True)
        best_sc, second_sc = s1_list[0][1], s1_list[1][1]
        cnt = float(len(s1_list))
        for rank, (sid, sc) in enumerate(s1_list):
            if rank == 0:
                rev[(sid, cid)] = [cnt, 1.0, float(best_sc - second_sc)]
            else:
                rev[(sid, cid)] = [cnt, 0.0, float(sc - best_sc)]
    return rev


def _featurize_s1(sid, s1_nums, s1_has_addr, cand_ids, base_feats, pool, rev):
    """43 features for one S1's candidates. Both sides use Silver fields only."""
    items = []
    for cid, f in zip(cand_ids, base_feats):
        items.append({
            'cid': cid,
            'comb_sim': f[10],
            'n_tsort': f[2],
            'brand': pool[cid]['brand'],
            'src': 'S2' if cid.startswith('S2-') else 'S3',
        })
    grp = compute_group_context_features(items)
    rows = []
    for i, cid in enumerate(cand_ids):
        f = base_feats[i]
        c = pool[cid]
        c_nums = c['nums']
        strict = 1.0 if (f[14] == 1.0 and f[2] < 0.80) else 0.0  # num_conflict & n_tsort < .8
        missing_one = 1.0 if (bool(s1_nums) != bool(c_nums)) else 0.0
        rows.append(
            list(f) + [strict, missing_one] + grp[i]
            + rev.get((sid, cid), UNCONTESTED_REV)
            + [0.0 if s1_has_addr else 1.0, 0.0 if c['has_addr'] else 1.0]
        )
    return rows


def build_universe(s1_path: str, s2_path: str, s3_path: str, out_dir: str,
                   top_k: Optional[int] = None, countries: Optional[List[str]] = None,
                   batch_size: int = 50000, log=print, resume: bool = True) -> dict:
    """
    For each country: full-pool blocking for every S1, reverse context over all S1 of the
    country, 43 features -> parquet chunks `feat_<country>_<b>.parquet` and candidate lists
    `cands_<country>.tsv` in out_dir. Returns a stats dict (also written to universe_stats.json).
    """
    top_k = top_k or config.UNIVERSE_TOP_K
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    bronze = BronzeIngestionEngine(threads=8)
    all_countries = [c for c, _ in bronze.get_test_countries(s1_path)]
    # explicit order if given (smallest first = earliest usable fallback), else file order
    todo = [c for c in countries if c in all_countries] if countries else all_countries
    stats_path = os.path.join(out_dir, "universe_stats.json")
    stats = {"top_k": top_k, "countries": {}}
    if resume and os.path.exists(stats_path):
        with open(stats_path) as f:
            prev = json.load(f)
        if prev.get("top_k") == top_k:
            stats = prev
    done = set(stats["countries"])
    t_all = time.time()

    for country in todo:
        if str(country) in done:
            log(f"[universe] {country}: already built, skipping (resume)")
            continue
        t0 = time.time()
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(country))
        s1_c, s23_c = bronze.load_country_data(s1_path, s2_path, s3_path, country)
        log(f"[universe] {country}: S1={len(s1_c):,} pool={len(s23_c):,}")

        pool = {}
        ids = s23_c['entity_id'].values
        names = s23_c['business_name'].values
        addrs = s23_c['business_address'].values
        ctry = s23_c['country'].values
        del s23_c
        for i in range(len(ids)):
            pool[ids[i]] = normalize_silver_entity(names[i], addrs[i])
        gold = GoldResolutionEngine(top_k=top_k)
        gold.build_index(ids, [pool[e] for e in ids], ctry)
        for d in pool.values():          # 'words' only feeds index keys; free it
            d.pop('words', None)
        del ids, names, addrs, ctry
        gc.collect()
        log(f"[universe] {country}: silver+index {time.time()-t0:.0f}s, keys={len(gold.index):,}")

        # Pass 1: blocking for every S1 (needed before reverse context can be computed)
        cand_to_s1s = defaultdict(list)
        n_batches = (len(s1_c) + batch_size - 1) // batch_size
        n_cands_total, n_empty = 0, 0
        cand_path = os.path.join(out_dir, f"cands_{safe}.tsv")
        with open(cand_path, "w", encoding="utf-8") as fc:
            for b in range(n_batches):
                chunk = []
                for row in s1_c.iloc[b * batch_size:(b + 1) * batch_size].itertuples():
                    sil = normalize_silver_entity(row.business_name, row.business_address)
                    cids, feats, _ = gold.query_and_rank_candidates(sil, country, pool)
                    fc.write(f"{row.entity_id}\t{','.join(cids)}\n")
                    n_cands_total += len(cids)
                    n_empty += (len(cids) == 0)
                    for cid, f in zip(cids, feats):
                        cand_to_s1s[cid].append((row.entity_id, f[10]))
                    chunk.append((row.entity_id, sil['nums'], sil['has_addr'], cids, feats))
                joblib.dump(chunk, os.path.join(tmp_dir, f"q_{safe}_{b}.joblib"), compress=1)
                del chunk
                done_n = min((b + 1) * batch_size, len(s1_c))
                el = time.time() - t0
                eta = el / done_n * (len(s1_c) - done_n)
                log(f"[universe] {country}: blocked {done_n:,}/{len(s1_c):,} "
                    f"elapsed {el/60:.1f}m, blocking ETA {eta/60:.1f}m (features pass adds ~30-50%)")
        del gold
        gc.collect()

        rev = _reverse_context(cand_to_s1s)
        del cand_to_s1s
        gc.collect()

        # Pass 2: features
        n_rows = 0
        for b in range(n_batches):
            qpath = os.path.join(tmp_dir, f"q_{safe}_{b}.joblib")
            chunk = joblib.load(qpath)
            os.remove(qpath)
            sids, cids_out, feats_out = [], [], []
            for sid, s1_nums, s1_has, cids, feats in chunk:
                if not cids:
                    continue
                rows = _featurize_s1(sid, s1_nums, s1_has, cids, feats, pool, rev)
                sids.extend([sid] * len(cids))
                cids_out.extend(cids)
                feats_out.extend(rows)
            if sids:
                df = pd.DataFrame(np.asarray(feats_out, dtype=np.float32), columns=FEATURE_NAMES)
                df.insert(0, 'sid', sids)
                df.insert(1, 'cid', cids_out)
                df.insert(2, 'is_s3', np.array([c.startswith('S3-') for c in cids_out], dtype=np.int8))
                df.insert(3, 'country', country)
                df.insert(4, 'bucket', np.array([s1_bucket(s) for s in sids], dtype=np.int16))
                df.to_parquet(os.path.join(out_dir, f"feat_{safe}_{b}.parquet"), index=False)
                n_rows += len(df)
            del chunk, sids, cids_out, feats_out
            gc.collect()

        n_s1 = len(s1_c)
        del pool, rev, s1_c
        gc.collect()
        stats["countries"][str(country)] = {
            "s1": n_s1,
            "pairs": n_rows,
            "mean_cands_per_s1": round(n_cands_total / max(1, n_s1), 3),
            "s1_with_no_cands": int(n_empty),
            "seconds": round(time.time() - t0, 1),
        }
        log(f"[universe] {country}: {n_rows:,} pairs in {time.time()-t0:.0f}s")
        with open(stats_path, "w") as f:   # checkpoint: country complete
            json.dump(stats, f, indent=2)

    if not os.listdir(tmp_dir):
        os.rmdir(tmp_dir)
    stats["seconds"] = round(time.time() - t_all, 1)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
