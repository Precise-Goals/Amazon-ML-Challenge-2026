"""
v3 test inference: same universe featurizer as training, fold-averaged LightGBM, isotonic
calibration, OOF-tuned per-source thresholds, global exclusivity, validator.

Countries absent from training (France) get `--unseen-offset` added to their thresholds
(open-set: decided from decision.json train_countries, nothing hard-coded). Default 0.0.

Usage (from code/business_entity_resolution):
  python src/predict_universe.py
  python src/predict_universe.py --skip-build --unseen-offset 0.05 --tag fr05
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
from universe import build_universe, FEATURE_NAMES
from decision_v3 import exclusive_best_mask, row_thresholds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=config.DATASET_DIR)
    ap.add_argument("--work-dir", default=os.path.join(config.BASE_DIR, "artifacts", "test_universe"))
    ap.add_argument("--model-dir", default=os.path.join(config.MODEL_DIR, "v3"))
    ap.add_argument("--out-dir", default=config.OUTPUT_DIR)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--unseen-offset", type=float, default=0.0)
    ap.add_argument("--th-s2", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--th-s3", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--tag", default="", help="suffix for a copy of the outputs, e.g. fr05")
    ap.add_argument("--build-only", action="store_true",
                    help="only build the test universe (no models needed); run in parallel with training")
    ap.add_argument("--rebuild", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    t0 = time.time()
    te = os.path.join(a.data_dir, "test")
    s1p, s2p, s3p = (os.path.join(te, f"test_source{i}.tsv") for i in (1, 2, 3))
    if a.build_only:
        build_universe(s1p, s2p, s3p, a.work_dir, top_k=config.UNIVERSE_TOP_K, resume=not a.rebuild)
        print("test universe built", flush=True)
        return
    with open(os.path.join(a.model_dir, "decision.json")) as f:
        dec = json.load(f)
    assert dec["features"] == FEATURE_NAMES, "model/featurizer mismatch: retrain"
    th2 = a.th_s2 if a.th_s2 is not None else dec["th_s2"]
    th3 = a.th_s3 if a.th_s3 is not None else dec["th_s3"]
    models = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(a.model_dir, "lgbm_fold*.txt")))]
    iso = joblib.load(os.path.join(a.model_dir, "isotonic.joblib"))
    print(f"models={len(models)} th_s2={th2} th_s3={th3} unseen_offset={a.unseen_offset}", flush=True)

    if not a.skip_build:
        build_universe(s1p, s2p, s3p, a.work_dir, top_k=dec.get("top_k"), resume=not a.rebuild)
    with open(os.path.join(a.work_dir, "universe_stats.json")) as f:
        ustats = json.load(f)
    n_test_ctry = len(BronzeIngestionEngine().get_test_countries(s1p))
    if len(ustats["countries"]) < n_test_ctry:
        raise SystemExit(f"test universe incomplete: {list(ustats['countries'])}; rerun without --skip-build")

    # score; keep only rows that pass their threshold (exclusivity among survivors is
    # identical to global exclusivity because a candidate's rows share one threshold)
    seen = set(dec["train_countries"])
    keep = []
    p_hist = np.zeros(20, dtype=np.int64)
    for p in sorted(glob.glob(os.path.join(a.work_dir, "feat_*.parquet"))):
        d = pd.read_parquet(p)
        X = d[FEATURE_NAMES].to_numpy(np.float32)
        pr = np.mean([m.predict(X) for m in models], axis=0)
        pc = iso.predict(pr).astype(np.float32)
        p_hist += np.histogram(pc, bins=20, range=(0, 1))[0]
        unseen = ~d["country"].isin(seen).to_numpy()
        th = row_thresholds(d["is_s3"].to_numpy(), th2, th3, unseen, a.unseen_offset)
        m = pc >= th
        keep.append(pd.DataFrame({"sid": d["sid"].to_numpy()[m], "cid": d["cid"].to_numpy()[m],
                                  "country": d["country"].to_numpy()[m], "p": pc[m]}))
    kept = pd.concat(keep, ignore_index=True)
    codes, _ = pd.factorize(kept["cid"])
    kept = kept[exclusive_best_mask(kept["p"].to_numpy(), codes)]
    matches = kept.sort_values("p", ascending=False).groupby("sid")["cid"].apply(list).to_dict()

    # outputs in exact test S1 order
    os.makedirs(a.out_dir, exist_ok=True)
    order = BronzeIngestionEngine().get_all_s1_ordered(s1p)
    cands = {}
    for cf in glob.glob(os.path.join(a.work_dir, "cands_*.tsv")):
        with open(cf, encoding="utf-8") as f:
            for line in f:
                sid, _, lst = line.rstrip("\n").partition("\t")
                cands[sid] = lst
    mpath = os.path.join(a.out_dir, "matching_results.tsv")
    cpath = os.path.join(a.out_dir, "candidate_pairs.tsv")
    n_links, n_empty, n_cand = 0, 0, 0
    with open(mpath, "w", encoding="utf-8", newline="\n") as fm, \
         open(cpath, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            ml = list(dict.fromkeys(matches.get(sid, [])))
            cl = cands.get(sid, "")
            fm.write(f"{sid}\t{','.join(ml)}\n")
            fc.write(f"{sid}\t{cl}\n")
            n_links += len(ml)
            n_empty += (len(ml) == 0)
            n_cand += len(cl.split(",")) if cl else 0

    per_country = kept.groupby("country").agg(links=("cid", "size"), s1_matched=("sid", "nunique")).to_dict("index")
    s1_per_country = {c: v["s1"] for c, v in ustats["countries"].items()}
    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": "v3_test_predict", "tag": a.tag,
        "th_s2": th2, "th_s3": th3, "unseen_offset": a.unseen_offset,
        "n_s1": len(order), "links": n_links, "mean_links_per_s1": round(n_links / len(order), 4),
        "pred_empty_rate": round(n_empty / len(order), 4),
        "mean_cands_per_s1": round(n_cand / len(order), 3),
        "per_country": {c: {**v, "s1": s1_per_country.get(c),
                            "links_per_s1": round(v["links"] / max(1, s1_per_country.get(c) or 1), 4),
                            "pred_empty_rate": round(1 - v["s1_matched"] / max(1, s1_per_country.get(c) or 1), 4)}
                        for c, v in per_country.items()},
        "p_cal_hist_20bins": p_hist.tolist(),
        "universe": ustats, "duration_sec": round(time.time() - t0, 1),
    }
    val = os.path.join(config.BASE_DIR, "datasource", "utils", "validate_submission.py")
    if os.path.exists(val):
        r = subprocess.run([sys.executable, val, "--matching", mpath, "--candidate", cpath,
                            "--test-dir", te], capture_output=True, text=True)
        stats["validator"] = (r.stdout + r.stderr).strip()[-2000:]
        stats["validator_pass"] = r.returncode == 0
    if a.tag:
        import shutil
        for pth in (mpath, cpath):
            shutil.copy(pth, pth.replace(".tsv", f"_{a.tag}.tsv"))
    os.makedirs(os.path.join(config.BASE_DIR, "reports"), exist_ok=True)
    with open(os.path.join(config.BASE_DIR, "reports", "experiments.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "universe"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
