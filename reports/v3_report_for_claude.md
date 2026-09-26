# v3 report for Claude

- python 3.14.5 on Windows-11-10.0.26200-SP0; lightgbm 4.7.0, sklearn 1.9.1
- LEADERBOARD SCORE of the submitted file(s): **<fill in: which tag, score>**

## Training (honest full-universe OOF)
```json
{
 "timestamp": "2026-09-26 18:50:54",
 "phase": "v3_full_universe",
 "run_id": "20260926-184448",
 "countries": [
  "India"
 ],
 "train_frac": 0.25,
 "n_s1": 883188,
 "blocking": {
  "pair_recall": 0.844,
  "entity_recall_ceiling": 0.6761,
  "mean_cands_per_s1": 11.242,
  "p95_cands_per_s1": 12.0
 },
 "oof": {
  "macro_f05": 0.8692921449882118,
  "micro_p": 0.9633042934610565,
  "micro_r": 0.7946097234400589,
  "singleton_acc": 0.8783611274340946,
  "mean_pred_per_s1": 2.8578298165283043,
  "k_buckets": {
   "0": 0.8784,
   "1": 0.7205,
   "2": 0.8372,
   "3": 0.8726,
   "4": 0.8878,
   "5": 0.8976,
   "6+": 0.9035
  },
  "country_f05": {
   "India": 0.8693
  }
 },
 "th_s2": 0.575,
 "th_s3": 0.575,
 "oof_no_exclusivity": 0.8685,
 "loco": {},
 "top_gain_features": [
  "gap_to_second_s1",
  "num_overlap",
  "is_mutual_best",
  "count_sim_gt_90",
  "num_conflict",
  "n_ratio",
  "a_tset",
  "legal_conflict",
  "max_sim",
  "n_tsort",
  "a_tsort",
  "comb_sim",
  "n_partial",
  "brand_len",
  "n_wratio"
 ],
 "duration_sec": 365.9
}
```

## Test predictions
```json
{
 "timestamp": "2026-09-27 00:49:16",
 "phase": "v3_test_predict",
 "tag": "base",
 "th_s2": 0.575,
 "th_s3": 0.575,
 "unseen_offset": 0.0,
 "n_s1": 1732544,
 "links": 5360729,
 "mean_links_per_s1": 3.0941,
 "pred_empty_rate": 0.0817,
 "mean_cands_per_s1": 11.156,
 "per_country": {
  "France": {
   "links": 809404,
   "s1_matched": 239961,
   "s1": 259452,
   "links_per_s1": 3.1197,
   "pred_empty_rate": 0.0751
  },
  "India": {
   "links": 2385296,
   "s1_matched": 727328,
   "s1": 809986,
   "links_per_s1": 2.9449,
   "pred_empty_rate": 0.102
  },
  "US": {
   "links": 2166029,
   "s1_matched": 623661,
   "s1": 663106,
   "links_per_s1": 3.2665,
   "pred_empty_rate": 0.0595
  }
 },
 "p_cal_hist_20bins": [
  12108039,
  521080,
  233273,
  176387,
  180286,
  132530,
  118701,
  122132,
  129844,
  100029,
  81226,
  106714,
  93860,
  78655,
  113362,
  91530,
  158476,
  218166,
  297841,
  4265697
 ],
 "duration_sec": 9688.3,
 "validator": "ML Challenge 2026 \u2014 submission validator\n  test dir: D:\\Workspace\\Projects\\ml2\\datasource\\dataset\\test\n  required S1 entities: 1732544\n  matching_results.tsv: 1732544 rows (141594 empty, 1590950 non-empty).\n  candidate_pairs.tsv: 1732544 rows (22744 empty, 1709800 non-empty).\n\nWARNING: ID-existence check is OFF (the default) \u2014 not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.\nPASS \u2014 no blocking issues found. Safe to submit.",
 "validator_pass": true
}
{
 "timestamp": "2026-09-27 01:03:22",
 "phase": "v3_test_predict",
 "tag": "unseen05",
 "th_s2": 0.575,
 "th_s3": 0.575,
 "unseen_offset": 0.05,
 "n_s1": 1732544,
 "links": 5320427,
 "mean_links_per_s1": 3.0709,
 "pred_empty_rate": 0.0829,
 "mean_cands_per_s1": 11.156,
 "per_country": {
  "France": {
   "links": 803124,
   "s1_matched": 239631,
   "s1": 259452,
   "links_per_s1": 3.0955,
   "pred_empty_rate": 0.0764
  },
  "India": {
   "links": 2385296,
   "s1_matched": 727328,
   "s1": 809986,
   "links_per_s1": 2.9449,
   "pred_empty_rate": 0.102
  },
  "US": {
   "links": 2132007,
   "s1_matched": 621929,
   "s1": 663106,
   "links_per_s1": 3.2152,
   "pred_empty_rate": 0.0621
  }
 },
 "p_cal_hist_20bins": [
  12108039,
  521080,
  233273,
  176387,
  180286,
  132530,
  118701,
  122132,
  129844,
  100029,
  81226,
  106714,
  93860,
  78655,
  113362,
  91530,
  158476,
  218166,
  297841,
  4265697
 ],
 "duration_sec": 804.9,
 "validator": "ML Challenge 2026 \u2014 submission validator\n  test dir: D:\\Workspace\\Projects\\ml2\\datasource\\dataset\\test\n  required S1 entities: 1732544\n  matching_results.tsv: 1732544 rows (143656 empty, 1588888 non-empty).\n  candidate_pairs.tsv: 1732544 rows (22744 empty, 1709800 non-empty).\n\nWARNING: ID-existence check is OFF (the default) \u2014 not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.\nPASS \u2014 no blocking issues found. Safe to submit.",
 "validator_pass": true
}
```

## Train log tail
```
2026-09-26 18:44:48,982 | universe: {"top_k": 12, "countries": {"India": {"s1": 883188, "pairs": 9929124, "mean_cands_per_s1": 11.242, "s1_with_no_cands": 17722, "seconds": 3317.6}}}
2026-09-26 18:44:49,640 | S1=883,188 links=3,059,843 singletons=0.0559
2026-09-26 18:44:57,635 | training rows=2,481,390 pos=0.2601 (train_frac=0.25)
2026-09-26 18:45:29,326 | fold model 0 trained on 1,986,785 rows
2026-09-26 18:46:01,350 | fold model 1 trained on 1,987,922 rows
2026-09-26 18:46:33,112 | fold model 2 trained on 1,985,358 rows
2026-09-26 18:47:03,572 | fold model 3 trained on 1,983,595 rows
2026-09-26 18:47:34,796 | fold model 4 trained on 1,981,900 rows
2026-09-26 18:48:51,684 | blocking: {'pair_recall': 0.844, 'entity_recall_ceiling': 0.6761, 'mean_cands_per_s1': 11.242, 'p95_cands_per_s1': 12.0}
2026-09-26 18:50:54,658 | OOF best th_s2=0.575 th_s3=0.575: {"macro_f05": 0.8692921449882118, "micro_p": 0.9633042934610565, "micro_r": 0.7946097234400589, "singleton_acc": 0.8783611274340946, "mean_pred_per_s1": 2.8578298165283043, "k_buckets": {"0": 0.8784, "1": 0.7205, "2": 0.8372, "3": 0.8726, "4": 0.8878, "5": 0.8976, "6+": 0.9035}, "country_f05": {"India": 0.8693}}
2026-09-26 18:50:54,658 | OOF without exclusivity at same th: 0.8685; top-10 grid: [(0.575, 0.575, 0.8692921449882118), (0.6, 0.575, 0.8692468864643046), (0.575, 0.6, 0.869242032160791), (0.625, 0.575, 0.8692338765602543), (0.575, 0.55, 0.8692039458599656), (0.575, 0.625, 0.8691968313259346), (0.6, 0.6, 0.8691953462251661), (0.625, 0.6, 0.8691805331104222), (0.6, 0.55, 0.869162794845352), (0.55, 0.575, 0.8691608704876662)]
2026-09-26 18:50:54,876 | done in 366s -> D:\Workspace\Projects\ml2\code\business_entity_resolution\src\..\models\v3
```
