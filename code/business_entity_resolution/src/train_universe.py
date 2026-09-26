"""
v3 training: honest, test-mirroring validation.

  1. build_universe() on the TRAIN files: every train S1 blocked against its country's FULL
     pool (same density/competition as test), features from the shared featurizer.
  2. Labels joined from GT with DuckDB (out-of-core).
  3. 5 fold models (fold = crc32(sid) % 5), each trained on a deterministic S1 sample of the
     other folds; OOF probabilities for EVERY train pair.
  4. Isotonic calibration on OOF; per-source threshold grid with global exclusivity, scored
     with the exact macro F0.5 over ALL train S1 (blocking misses count as FN).
  5. Leave-one-country-out check (proxy for unseen countries such as France).

Usage (from code/business_entity_resolution):
  python src/train_universe.py                         # full train universe (final numbers)
  python src/train_universe.py --countries India       # fast honest iteration on one country
  python src/train_universe.py --skip-build            # reuse the built universe
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
import numpy as np
import duckdb
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from universe import build_universe, FEATURE_NAMES
from decision_v3 import exclusive_best_mask, row_thresholds, summarize, f05_per_s1

S3_OFFSET = 10 ** 12


def get_logger(path):
    log = logging.getLogger("train_v3")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log.info


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=config.DATASET_DIR)
    ap.add_argument("--work-dir", default=os.path.join(config.BASE_DIR, "artifacts", "train_universe"))
    ap.add_argument("--model-dir", default=os.path.join(config.MODEL_DIR, "v3"))
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.25,
                    help="fraction of S1 (per fold complement) used to fit each fold model")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--save-oof", action="store_true")
    return ap.parse_args()


def id_num_sql(col):
    return (f"(TRY_CAST(substr({col}, 4) AS BIGINT) + "
            f"CASE WHEN {col} LIKE 'S3-%' THEN {S3_OFFSET} ELSE 0 END)")


def main():
    a = parse_args()
    os.makedirs(a.work_dir, exist_ok=True)
    os.makedirs(a.model_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log = get_logger(os.path.join(a.work_dir, f"train_{run_id}.log"))
    t0 = time.time()
    tr = os.path.join(a.data_dir, "train")
    s1p, s2p, s3p = (os.path.join(tr, f"train_source{i}.tsv") for i in (1, 2, 3))
    gtp = os.path.join(tr, "train_ground_truth.tsv")

    # ---------- 1. Universe ----------
    if not a.skip_build or not glob.glob(os.path.join(a.work_dir, "feat_*.parquet")):
        for f in glob.glob(os.path.join(a.work_dir, "feat_*.parquet")):
            os.remove(f)
        ustats = build_universe(s1p, s2p, s3p, a.work_dir, countries=a.countries, log=log)
    else:
        with open(os.path.join(a.work_dir, "universe_stats.json")) as f:
            ustats = json.load(f)
    log(f"universe: {json.dumps(ustats)}")

    # ---------- 2. GT / S1 table ----------
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    rd = "delim='\\t', header=true, all_varchar=true"
    ctry_filter = ""
    if a.countries:
        ctry_filter = "WHERE s.country IN (" + ",".join(f"'{c}'" for c in a.countries) + ")"
    s1_tab = con.execute(f"""
        SELECT {id_num_sql('s.entity_id')} AS sid_num, s.country,
               CASE WHEN g.matched_entity_ids IS NULL OR trim(g.matched_entity_ids) = '' THEN 0
                    ELSE len(string_split(g.matched_entity_ids, ',')) END AS ntrue
        FROM read_csv('{s1p}', {rd}) s
        LEFT JOIN read_csv('{gtp}', {rd}) g ON g.source1_entity_id = s.entity_id
        {ctry_filter}
        ORDER BY sid_num""").df()
    if s1_tab["sid_num"].isna().any():
        raise ValueError("Non-numeric entity_id suffix found; extend id_num_sql encoding.")
    s1_sorted = s1_tab["sid_num"].to_numpy(np.int64)
    ntrue = s1_tab["ntrue"].to_numpy(np.int64)
    s1_country = s1_tab["country"].to_numpy()
    log(f"S1={len(s1_sorted):,} links={ntrue.sum():,} singletons={(ntrue == 0).mean():.4f}")
    con.execute(f"""
        CREATE TABLE links AS
        SELECT {id_num_sql('source1_entity_id')} AS s, {id_num_sql('c')} AS c FROM (
          SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS c
          FROM read_csv('{gtp}', {rd})
          WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> '')""")

    files = sorted(glob.glob(os.path.join(a.work_dir, "feat_*.parquet")))
    feat_sql = ", ".join(f'f."{c}"' for c in FEATURE_NAMES)

    def load(path, where="TRUE"):
        return con.execute(f"""
            SELECT {id_num_sql('f.sid')} AS sid_num, {id_num_sql('f.cid')} AS cid_num,
                   f.is_s3, f.country, f.bucket, (l.c IS NOT NULL) AS label, {feat_sql}
            FROM read_parquet('{path}') f
            LEFT JOIN links l ON l.s = {id_num_sql('f.sid')} AND l.c = {id_num_sql('f.cid')}
            WHERE {where}""").df()

    # ---------- 3. Fold models ----------
    cut = int(round(a.train_frac * 200))  # bucket // 5 in [0, 200)
    Xs, ys, folds, ctrs = [], [], [], []
    for p in files:
        d = load(p, f"f.bucket // 5 < {cut}")
        Xs.append(d[FEATURE_NAMES].to_numpy(np.float32))
        ys.append(d["label"].to_numpy(np.int8))
        folds.append((d["bucket"].to_numpy() % 5).astype(np.int8))
        ctrs.append(d["country"].to_numpy())
    X = np.concatenate(Xs); y = np.concatenate(ys); fold = np.concatenate(folds); ctr = np.concatenate(ctrs)
    del Xs, ys, folds, ctrs
    log(f"training rows={len(y):,} pos={y.mean():.4f} (train_frac={a.train_frac})")

    params = dict(objective="binary", learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                  min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, num_threads=0)
    models = []
    for k in range(5):
        m = fold != k
        ds = lgb.Dataset(X[m], y[m], feature_name=FEATURE_NAMES, free_raw_data=True)
        models.append(lgb.train(params, ds, num_boost_round=a.n_estimators))
        log(f"fold model {k} trained on {m.sum():,} rows")

    # ---------- 4. OOF for every pair ----------
    ctry_code = {c: i for i, c in enumerate(sorted(set(s1_country)))}
    P, L, SID, CID, S3, CT = [], [], [], [], [], []
    for p in files:
        d = load(p)
        fk = (d["bucket"].to_numpy() % 5)
        Xa = d[FEATURE_NAMES].to_numpy(np.float32)
        pr = np.empty(len(d), dtype=np.float32)
        for k in range(5):
            mk = fk == k
            if mk.any():
                pr[mk] = models[k].predict(Xa[mk])
        P.append(pr); L.append(d["label"].to_numpy(bool))
        SID.append(d["sid_num"].to_numpy(np.int64)); CID.append(d["cid_num"].to_numpy(np.int64))
        S3.append(d["is_s3"].to_numpy(np.int8))
        CT.append(d["country"].map(ctry_code).to_numpy(np.int16))
    p_raw = np.concatenate(P); label = np.concatenate(L)
    sid_num = np.concatenate(SID); cid_num = np.concatenate(CID)
    is_s3 = np.concatenate(S3); row_ctry = np.concatenate(CT)
    del P, L, SID, CID, S3, CT
    sid_codes = np.searchsorted(s1_sorted, sid_num)
    assert (s1_sorted[sid_codes] == sid_num).all(), "pair S1 not found in S1 table"
    _, cid_codes = np.unique(cid_num, return_inverse=True)

    # blocking report
    n = len(s1_sorted)
    tp_all = np.bincount(sid_codes, weights=label, minlength=n)
    ncand = np.bincount(sid_codes, minlength=n)
    blocking = {
        "pair_recall": round(float(tp_all.sum() / max(1, ntrue.sum())), 4),
        "entity_recall_ceiling": round(float((tp_all >= ntrue).mean()), 4),
        "mean_cands_per_s1": round(float(ncand.mean()), 3),
        "p95_cands_per_s1": float(np.percentile(ncand, 95)),
    }
    log(f"blocking: {blocking}")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(p_raw), size=min(len(p_raw), 5_000_000), replace=False)
    iso.fit(p_raw[fit_idx], label[fit_idx].astype(np.float64))
    p_cal = iso.predict(p_raw).astype(np.float32)
    best_mask = exclusive_best_mask(p_cal, cid_codes)

    grid = np.round(np.arange(0.30, 0.951, 0.025), 3)
    best = (-1.0, None, None)
    table = []
    for t2 in grid:
        for t3 in grid:
            th = row_thresholds(is_s3, t2, t3)
            pred = (p_cal >= th) & best_mask
            f = f05_per_s1(sid_codes, pred, label, ntrue).mean()
            table.append((float(t2), float(t3), float(f)))
            if f > best[0]:
                best = (float(f), float(t2), float(t3))
    _, th2, th3 = best
    pred = (p_cal >= row_thresholds(is_s3, th2, th3)) & best_mask
    oof = summarize(sid_codes, pred, label, ntrue, s1_country)
    # robustness: how flat is the optimum? (a sharp peak = fragile on test)
    near = sorted(table, key=lambda r: -r[2])[:10]
    no_excl = f05_per_s1(sid_codes, p_cal >= row_thresholds(is_s3, th2, th3), label, ntrue).mean()
    log(f"OOF best th_s2={th2} th_s3={th3}: {json.dumps(oof)}")
    log(f"OOF without exclusivity at same th: {no_excl:.4f}; top-10 grid: {near}")

    # ---------- 5. LOCO ----------
    loco = {}
    train_countries = sorted(set(ctr))
    if not a.no_loco and len(train_countries) >= 2:
        for held in train_countries:
            m = ctr != held
            mdl = lgb.train(params, lgb.Dataset(X[m], y[m], feature_name=FEATURE_NAMES),
                            num_boost_round=a.n_estimators)
            rows = row_ctry == ctry_code[held]
            # re-read held-out rows in the same file/row order as the OOF arrays
            parts = []
            for p in files:
                d = load(p)
                parts.append(d.loc[d["country"] == held, FEATURE_NAMES].to_numpy(np.float32))
            Xh = np.concatenate(parts)
            ph = iso.predict(mdl.predict(Xh)).astype(np.float32)
            p_mix = p_cal.copy()
            p_mix[rows] = ph
            bm = exclusive_best_mask(p_mix, cid_codes)
            pr = (p_mix >= row_thresholds(is_s3, th2, th3)) & bm
            fs = f05_per_s1(sid_codes, pr, label, ntrue)
            sel = s1_country == held
            loco[str(held)] = {"loco_f05": round(float(fs[sel].mean()), 4),
                               "in_dist_oof_f05": oof.get("country_f05", {}).get(str(held))}
            log(f"LOCO held-out {held}: {loco[str(held)]}")
    del X, y

    # ---------- 6. Save ----------
    for k, mdl in enumerate(models):
        mdl.save_model(os.path.join(a.model_dir, f"lgbm_fold{k}.txt"))
    joblib.dump(iso, os.path.join(a.model_dir, "isotonic.joblib"))
    decision = {"th_s2": th2, "th_s3": th3, "train_countries": train_countries,
                "features": FEATURE_NAMES, "top_k": ustats.get("top_k"), "run_id": run_id}
    with open(os.path.join(a.model_dir, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    imp = sorted(zip(FEATURE_NAMES, models[0].feature_importance("gain")), key=lambda r: -r[1])
    if a.save_oof:
        import pandas as pd
        pd.DataFrame({"sid_num": sid_num, "cid_num": cid_num, "p_cal": p_cal, "label": label,
                      "pred": pred}).to_parquet(os.path.join(a.work_dir, "oof.parquet"), index=False)

    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": "v3_full_universe",
             "run_id": run_id, "countries": a.countries or "all", "train_frac": a.train_frac,
             "n_s1": int(n), "blocking": blocking, "oof": oof, "th_s2": th2, "th_s3": th3,
             "oof_no_exclusivity": round(float(no_excl), 4), "loco": loco,
             "top_gain_features": [r[0] for r in imp[:15]],
             "duration_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(config.BASE_DIR, "reports"), exist_ok=True)
    with open(os.path.join(config.BASE_DIR, "reports", "experiments.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"done in {time.time()-t0:.0f}s -> {a.model_dir}")


if __name__ == "__main__":
    main()
