# v3 runbook: honest validation + no train/test skew

## Why 0.776 on the leaderboard vs 0.964 locally
1. **The validation universe did not look like test.** `bronze.load_train_sample` took 50k S1 (2.3% of train). Their pool held **all** their true matches but only a 250k random sample of each of S2/S3 (~5%). On test, every S1 competes with all 10M records, including the true matches of 1.73M other businesses (chains, co-tenants, near-duplicates). Locally, that competition was about 20× weaker:
   - `cand_s1_count`, `is_mutual_best`, `gap_to_second_s1` were almost always "uncontested" in training.
   - The rank/gap features saw easy neighbourhoods.
   - Isotonic calibration and thresholds were fitted where positives were over-represented.
   - Result: probabilities that are too high on test, so false merges.
2. **Train/test feature skew in v2**, where the same feature was computed differently:
   | feature | train (train_phase_b) | test (predict_submission) |
   |---|---|---|
   | agree_max_other / agree_mean_other / count_sim_gt_90 | raw lower-cased name | Silver brand (suffix stripped, transliterated) |
   | strict_num_conflict | raw S1 name vs raw cand name | raw S1 name vs Silver cand brand |
   | c_nums, num_missing_one_side | regex on raw address (keeps ZIP) | Silver street nums (ZIP removed) |
   | c_addr_missing | NaN address becomes "nan", so never missing | Silver has_addr (3.3% missing) |
   | top_k (n_cands, n_s2, n_s3, ranks) | 15 | 12 |

## What v3 changes
- `src/universe.py`: **one featurizer for train and test.** Every S1 of a country is blocked against that country's full pool. Reverse context is computed over all S1. The 43 features (same names and order as v2) come from Silver fields only, with the same top_k (12) on both sides.
- `src/train_universe.py`: runs the universe on train, joins labels with DuckDB, then trains 5 fold models (fold = crc32(sid) % 5).
  - Out-of-fold (OOF) predictions cover **every** train pair.
  - Isotonic calibration, then a per-source threshold grid with global exclusivity, scored with the exact macro F0.5 over **all** train S1 (blocking misses count).
  - Leave-one-country-out (LOCO) US↔India as a proxy for France.
- `src/predict_universe.py`: same featurizer on test, fold-averaged, calibrated, OOF-tuned thresholds and exclusivity, then the validator.
  - `--unseen-offset` raises thresholds only for countries absent from train (France), chosen from `decision.json`, not hard-coded.
- `src/decision_v3.py`: vectorized decision + metric, cross-checked against `eval_harness.evaluate_metrics` (identical to 6 decimals).
- Smoke-tested end to end on synthetic data (train → predict → validator PASS). **It has not been run on the real data yet.**

## Run (from repo root, where datasource/ lives)
```powershell
git fetch; git checkout worktree-fix-train-test-skew
powershell -ExecutionPolicy Bypass -File code/business_entity_resolution/scripts/run_v3.ps1
```
It produces `output/matching_results_base.tsv`, `output/matching_results_unseen05.tsv` (plus candidate files), and **`reports/v3_report_for_claude.md`**. Send that one file back. It contains aggregates only.

Short on time? First run `python code/business_entity_resolution/src/train_universe.py --countries India`. It is honest (one country, full pool) and much faster, and shows whether the OOF drops from 0.96 to a realistic level.

If RAM runs out on the biggest country (train US pool is about 6M records), send the console log. The next step would be sharding the Silver pool dict.

## Submission plan (5 left)
1. **Sub 1: `matching_results_base.tsv`.** Its LB score, together with the honest OOF on US/India, gives the France score by subtraction: LB ≈ 0.85·OOF(US,IN) + 0.15·F(France).
2. **Sub 2: `unseen05`**, but only if the implied France score is well below US/India. This is the one LB probe that is justified, because France has no labels.
3. Subs 3–4: only changes that improve the honest OOF by ≥ 0.003 without hurting LOCO (next candidates: char-ngram blocker, expected-F0.5 top-k decision).
4. Keep one submission in reserve. The final choice goes by OOF + LOCO, not by the public LB.
