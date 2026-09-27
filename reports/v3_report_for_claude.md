# v3 report for Claude

- python 3.14.5 on Windows-11-10.0.26200-SP0; lightgbm 4.7.0, sklearn 1.9.1
- LEADERBOARD SCORE of the submitted file(s): **0.880 (unified_unseen05 / matching_results.tsv)**

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
{
 "timestamp": "2026-09-27 14:27:32",
 "phase": "v3_full_universe",
 "run_id": "20260927-103428",
 "countries": [
  "India",
  "US"
 ],
 "train_frac": 0.25,
 "n_s1": 2206821,
 "blocking": {
  "pair_recall": 0.9002,
  "entity_recall_ceiling": 0.7683,
  "mean_cands_per_s1": 11.305,
  "p95_cands_per_s1": 12.0
 },
 "oof": {
  "macro_f05": 0.9086766766106807,
  "micro_p": 0.9683801904049587,
  "micro_r": 0.8529491586222968,
  "singleton_acc": 0.8954700722938489,
  "mean_pred_per_s1": 3.0486709162183976,
  "k_buckets": {
   "0": 0.8955,
   "1": 0.7921,
   "2": 0.8873,
   "3": 0.9132,
   "4": 0.9239,
   "5": 0.9307,
   "6+": 0.9351
  },
  "country_f05": {
   "India": 0.8642,
   "US": 0.9383
  }
 },
 "th_s2": 0.6,
 "th_s3": 0.575,
 "oof_no_exclusivity": 0.9081,
 "loco": {
  "India": {
   "loco_f05": 0.8272,
   "in_dist_oof_f05": 0.8642
  },
  "US": {
   "loco_f05": 0.9363,
   "in_dist_oof_f05": 0.9383
  }
 },
 "top_gain_features": [
  "gap_to_second_s1",
  "num_conflict",
  "is_mutual_best",
  "num_overlap",
  "legal_conflict",
  "count_sim_gt_90",
  "a_tset",
  "comb_sim",
  "n_ratio",
  "n_tsort",
  "a_tsort",
  "max_sim",
  "a_partial",
  "zip_conflict",
  "n_wratio"
 ],
 "duration_sec": 13983.8
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
{
 "timestamp": "2026-09-27 15:38:24",
 "phase": "v3_test_predict",
 "tag": "unified_base",
 "th_s2": 0.6,
 "th_s3": 0.575,
 "unseen_offset": 0.0,
 "n_s1": 1732544,
 "links": 5426872,
 "mean_links_per_s1": 3.1323,
 "pred_empty_rate": 0.0815,
 "mean_cands_per_s1": 11.156,
 "per_country": {
  "France": {
   "links": 816402,
   "s1_matched": 240063,
   "s1": 259452,
   "links_per_s1": 3.1466,
   "pred_empty_rate": 0.0747
  },
  "India": {
   "links": 2419089,
   "s1_matched": 727166,
   "s1": 809986,
   "links_per_s1": 2.9866,
   "pred_empty_rate": 0.1022
  },
  "US": {
   "links": 2191381,
   "s1_matched": 624042,
   "s1": 663106,
   "links_per_s1": 3.3047,
   "pred_empty_rate": 0.0589
  }
 },
 "p_cal_hist_20bins": [
  12145531,
  468030,
  295621,
  140878,
  189038,
  90721,
  122516,
  108167,
  85528,
  74163,
  120656,
  69644,
  83721,
  66654,
  100537,
  97583,
  138331,
  124701,
  329094,
  4476714
 ],
 "duration_sec": 1049.1,
 "validator": "ML Challenge 2026 \u2014 submission validator\n  test dir: D:\\Workspace\\Projects\\ml2\\datasource\\dataset\\test\n  required S1 entities: 1732544\n  matching_results.tsv: 1732544 rows (141273 empty, 1591271 non-empty).\n  candidate_pairs.tsv: 1732544 rows (22744 empty, 1709800 non-empty).\n\nWARNING: ID-existence check is OFF (the default) \u2014 not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.\nPASS \u2014 no blocking issues found. Safe to submit.",
 "validator_pass": true
}
{
 "timestamp": "2026-09-27 16:01:06",
 "phase": "v3_test_predict",
 "tag": "unified_unseen05",
 "th_s2": 0.6,
 "th_s3": 0.575,
 "unseen_offset": 0.05,
 "n_s1": 1732544,
 "links": 5419406,
 "mean_links_per_s1": 3.128,
 "pred_empty_rate": 0.0818,
 "mean_cands_per_s1": 11.156,
 "per_country": {
  "France": {
   "links": 808936,
   "s1_matched": 239696,
   "s1": 259452,
   "links_per_s1": 3.1179,
   "pred_empty_rate": 0.0761
  },
  "India": {
   "links": 2419089,
   "s1_matched": 727166,
   "s1": 809986,
   "links_per_s1": 2.9866,
   "pred_empty_rate": 0.1022
  },
  "US": {
   "links": 2191381,
   "s1_matched": 624042,
   "s1": 663106,
   "links_per_s1": 3.3047,
   "pred_empty_rate": 0.0589
  }
 },
 "p_cal_hist_20bins": [
  12145531,
  468030,
  295621,
  140878,
  189038,
  90721,
  122516,
  108167,
  85528,
  74163,
  120656,
  69644,
  83721,
  66654,
  100537,
  97583,
  138331,
  124701,
  329094,
  4476714
 ],
 "duration_sec": 1300.7,
 "validator": "ML Challenge 2026 \u2014 submission validator\n  test dir: D:\\Workspace\\Projects\\ml2\\datasource\\dataset\\test\n  required S1 entities: 1732544\n  matching_results.tsv: 1732544 rows (141640 empty, 1590904 non-empty).\n  candidate_pairs.tsv: 1732544 rows (22744 empty, 1709800 non-empty).\n\nWARNING: ID-existence check is OFF (the default) \u2014 not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.\nPASS \u2014 no blocking issues found. Safe to submit.",
 "validator_pass": true
}
```

## Train log tail
```
2026-09-27 10:34:28,719 | [universe] India: already built, skipping (resume)
2026-09-27 10:34:34,115 | [universe] US: S1=1,323,633 pool=6,186,873
2026-09-27 10:49:22,257 | [universe] US: silver+index 894s, keys=16,623,646
2026-09-27 10:55:12,128 | [universe] US: blocked 50,000/1,323,633 elapsed 20.7m, blocking ETA 527.9m (features pass adds ~30-50%)
2026-09-27 10:58:45,535 | [universe] US: blocked 100,000/1,323,633 elapsed 24.3m, blocking ETA 297.1m (features pass adds ~30-50%)
2026-09-27 11:02:30,604 | [universe] US: blocked 150,000/1,323,633 elapsed 28.0m, blocking ETA 219.3m (features pass adds ~30-50%)
2026-09-27 11:08:19,970 | [universe] US: blocked 200,000/1,323,633 elapsed 33.9m, blocking ETA 190.2m (features pass adds ~30-50%)
2026-09-27 11:12:55,010 | [universe] US: blocked 250,000/1,323,633 elapsed 38.4m, blocking ETA 165.1m (features pass adds ~30-50%)
2026-09-27 11:17:17,383 | [universe] US: blocked 300,000/1,323,633 elapsed 42.8m, blocking ETA 146.1m (features pass adds ~30-50%)
2026-09-27 11:21:23,496 | [universe] US: blocked 350,000/1,323,633 elapsed 46.9m, blocking ETA 130.5m (features pass adds ~30-50%)
2026-09-27 11:29:16,540 | [universe] US: blocked 400,000/1,323,633 elapsed 54.8m, blocking ETA 126.5m (features pass adds ~30-50%)
2026-09-27 11:34:17,320 | [universe] US: blocked 450,000/1,323,633 elapsed 59.8m, blocking ETA 116.1m (features pass adds ~30-50%)
2026-09-27 11:39:33,699 | [universe] US: blocked 500,000/1,323,633 elapsed 65.1m, blocking ETA 107.2m (features pass adds ~30-50%)
2026-09-27 11:44:32,634 | [universe] US: blocked 550,000/1,323,633 elapsed 70.1m, blocking ETA 98.6m (features pass adds ~30-50%)
2026-09-27 11:49:34,769 | [universe] US: blocked 600,000/1,323,633 elapsed 75.1m, blocking ETA 90.6m (features pass adds ~30-50%)
2026-09-27 11:57:35,729 | [universe] US: blocked 650,000/1,323,633 elapsed 83.1m, blocking ETA 86.1m (features pass adds ~30-50%)
2026-09-27 12:02:19,244 | [universe] US: blocked 700,000/1,323,633 elapsed 87.8m, blocking ETA 78.3m (features pass adds ~30-50%)
2026-09-27 12:07:05,271 | [universe] US: blocked 750,000/1,323,633 elapsed 92.6m, blocking ETA 70.8m (features pass adds ~30-50%)
2026-09-27 12:11:18,831 | [universe] US: blocked 800,000/1,323,633 elapsed 96.8m, blocking ETA 63.4m (features pass adds ~30-50%)
2026-09-27 12:15:05,885 | [universe] US: blocked 850,000/1,323,633 elapsed 100.6m, blocking ETA 56.1m (features pass adds ~30-50%)
2026-09-27 12:22:00,784 | [universe] US: blocked 900,000/1,323,633 elapsed 107.5m, blocking ETA 50.6m (features pass adds ~30-50%)
2026-09-27 12:26:20,774 | [universe] US: blocked 950,000/1,323,633 elapsed 111.9m, blocking ETA 44.0m (features pass adds ~30-50%)
2026-09-27 12:30:31,988 | [universe] US: blocked 1,000,000/1,323,633 elapsed 116.1m, blocking ETA 37.6m (features pass adds ~30-50%)
2026-09-27 12:34:06,964 | [universe] US: blocked 1,050,000/1,323,633 elapsed 119.6m, blocking ETA 31.2m (features pass adds ~30-50%)
2026-09-27 12:39:58,757 | [universe] US: blocked 1,100,000/1,323,633 elapsed 125.5m, blocking ETA 25.5m (features pass adds ~30-50%)
2026-09-27 12:46:06,000 | [universe] US: blocked 1,150,000/1,323,633 elapsed 131.6m, blocking ETA 19.9m (features pass adds ~30-50%)
2026-09-27 12:56:42,636 | [universe] US: blocked 1,200,000/1,323,633 elapsed 142.2m, blocking ETA 14.7m (features pass adds ~30-50%)
2026-09-27 13:05:24,777 | [universe] US: blocked 1,250,000/1,323,633 elapsed 150.9m, blocking ETA 8.9m (features pass adds ~30-50%)
2026-09-27 13:10:51,641 | [universe] US: blocked 1,300,000/1,323,633 elapsed 156.4m, blocking ETA 2.8m (features pass adds ~30-50%)
2026-09-27 13:13:29,120 | [universe] US: blocked 1,323,633/1,323,633 elapsed 159.0m, blocking ETA 0.0m (features pass adds ~30-50%)
2026-09-27 14:04:00,873 | [universe] US: 15,018,737 pairs in 12572s
2026-09-27 14:04:01,021 | universe: {"top_k": 12, "countries": {"India": {"s1": 883188, "pairs": 9929124, "mean_cands_per_s1": 11.242, "s1_with_no_cands": 17722, "seconds": 3317.6}, "US": {"s1": 1323633, "pairs": 15018737, "mean_cands_per_s1": 11.347, "s1_with_no_cands": 743, "seconds": 12572.1}}, "seconds": 12572.2}
2026-09-27 14:04:02,778 | S1=2,206,821 links=7,638,365 singletons=0.0558
2026-09-27 14:04:23,984 | training rows=6,222,791 pos=0.2757 (train_frac=0.25)
2026-09-27 14:06:07,651 | fold model 0 trained on 4,980,347 rows
2026-09-27 14:07:50,784 | fold model 1 trained on 4,982,633 rows
2026-09-27 14:09:38,012 | fold model 2 trained on 4,980,396 rows
2026-09-27 14:11:18,909 | fold model 3 trained on 4,971,490 rows
2026-09-27 14:13:03,492 | fold model 4 trained on 4,976,298 rows
2026-09-27 14:16:23,246 | blocking: {'pair_recall': 0.9002, 'entity_recall_ceiling': 0.7683, 'mean_cands_per_s1': 11.305, 'p95_cands_per_s1': 12.0}
2026-09-27 14:21:54,866 | OOF best th_s2=0.6 th_s3=0.575: {"macro_f05": 0.9086766766106807, "micro_p": 0.9683801904049587, "micro_r": 0.8529491586222968, "singleton_acc": 0.8954700722938489, "mean_pred_per_s1": 3.0486709162183976, "k_buckets": {"0": 0.8955, "1": 0.7921, "2": 0.8873, "3": 0.9132, "4": 0.9239, "5": 0.9307, "6+": 0.9351}, "country_f05": {"India": 0.8642, "US": 0.9383}}
2026-09-27 14:21:54,867 | OOF without exclusivity at same th: 0.9081; top-10 grid: [(0.6, 0.575, 0.9086766766106807), (0.625, 0.575, 0.9086623103047468), (0.6, 0.6, 0.9086526449493918), (0.625, 0.6, 0.9086332435720241), (0.6, 0.55, 0.9086312202358738), (0.625, 0.55, 0.9086196487219967), (0.65, 0.575, 0.9085922397539564), (0.575, 0.575, 0.9085879912386977), (0.575, 0.6, 0.9085666101928452), (0.65, 0.6, 0.9085631459900849)]
2026-09-27 14:24:48,270 | LOCO held-out India: {'loco_f05': 0.8272, 'in_dist_oof_f05': 0.8642}
2026-09-27 14:27:32,124 | LOCO held-out US: {'loco_f05': 0.9363, 'in_dist_oof_f05': 0.9383}
2026-09-27 14:27:32,312 | done in 13984s -> D:\Workspace\Projects\ml2\code\business_entity_resolution\src\..\models\v3
```
