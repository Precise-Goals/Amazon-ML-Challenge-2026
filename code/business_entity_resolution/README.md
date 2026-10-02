# Business Entity Resolution Pipeline — Team Falcons

High-performance, machine learning-driven Business Entity Resolution pipeline engineered for large-scale commercial entity deduplication and cross-source record linkage across millions of noisy records.

**Official Leaderboard Score: `0.880` (Macro F₀.₅)**

---

## 1. Overview & Architecture

Given business records across three independent sources (`Source 1`, `Source 2`, and `Source 3`):
- **Source 1** is the deduplicated reference source.
- The pipeline discovers all matching entities from **Source 2** and **Source 3** for each **Source 1** entity.
- Evaluated using **Macro-averaged F₀.₅**, which weights precision 2× over recall.

### Pipeline Architecture (Medallion):

```
  [ Bronze Layer ]          Zero-copy DuckDB streaming ingestion, country partitioning
        │
        ▼
  [ Silver Layer ]          Unicode normalisation, Devanagari→Latin transliteration,
        │                   legal form extraction, address normalisation & expansion
        ▼
  [ Gold Layer  ]
        │
        ├─ Multi-Pass Inverted Index Blocking
        │    Keys: tok, bigram, pref4, addr_tok, addr_pair,
        │          zip_num, sn_w, zip_w, bn_sn, bn_zip, skel
        │
        ├─ 43-Feature Pair Feature Extraction
        │    (Name/Address RapidFuzz sims, discrete anchors,
        │     Group Context + Reverse Context features)
        │
        ├─ 5-Fold LightGBM Ensemble (Unified India + US Training)
        │    + Isotonic Probability Calibration
        │
        ├─ Country-Aware Decision Thresholds
        │    (Base + conservative +0.05 offset for unseen France)
        │
        └─ Global Greedy Bipartite Exclusivity Layer
             │
             ├── matching_results.tsv   (Final predicted matches)
             └── candidate_pairs.tsv   (Blocking candidate set)
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py              # Package interface
│   ├── config.py                # Hyperparameters, file paths, stopping rules
│   ├── utils.py                 # Text normalizers, address parser, macro F₀.₅ scorer
│   ├── bronze.py                # Bronze Layer: zero-copy DuckDB ingestion & country partition
│   ├── silver.py                # Silver Layer: normalisation, Devanagari transliteration, legal forms
│   ├── gold.py                  # Gold Layer: multi-pass blocking, dense features, precision gates
│   ├── universe.py              # Universe-scale index builder (contiguous int32 for 6.2M US pool)
│   ├── features.py              # RapidFuzz feature extraction (legacy single-country)
│   ├── features_v2.py           # Group context, reverse context, and agreement features
│   ├── train_universe.py        # ★ Unified 5-fold LightGBM training (India + US, 2.2M entities)
│   ├── predict_universe.py      # ★ Full-universe inference with country-aware thresholds
│   ├── train_phase_b.py         # Phase B single-country LightGBM trainer (reference)
│   ├── phase_c_decision.py      # Phase C decision boundary optimiser
│   ├── phase_d_e_blocking.py    # Phase D/E blocking & candidate pruner
│   ├── predict_submission.py    # Legacy single-country inference runner
│   ├── eval_harness.py          # Offline evaluation harness (macro F₀.₅ on labelled data)
│   ├── cache_and_tune.py        # Hyperparameter search & caching utilities
│   ├── decision_v3.py           # V3 decision boundary logic
│   └── sweep_thresholds.py      # Threshold sweep utilities
├── models/
│   └── v3/
│       ├── lgbm_fold0.txt       # Fold-0 LightGBM model (native format)
│       ├── lgbm_fold1.txt       # Fold-1 LightGBM model
│       ├── lgbm_fold2.txt       # Fold-2 LightGBM model
│       ├── lgbm_fold3.txt       # Fold-3 LightGBM model
│       └── lgbm_fold4.txt       # Fold-4 LightGBM model
├── decision.json                # Calibrated per-source decision thresholds
├── README.md                    # This file
└── requirements.txt             # Pinned Python package dependencies
```

---

## 3. Environment Setup

### Prerequisites
- **Python 3.10+** (tested on 3.11 and 3.12)
- **RAM:** 16 GB minimum; **24 GB recommended** for full US pool (6.2M entities)
- **Storage:** ~5 GB free for intermediate files

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Data Layout

The pipeline expects the competition data laid out as follows relative to the **repository root** (`ml2/`):

```
datasource/
└── dataset/
    ├── train/
    │   ├── source1.csv        # Training Source 1 records
    │   ├── source2.csv        # Training Source 2 records
    │   ├── source3.csv        # Training Source 3 records
    │   └── gt_links.csv       # Ground-truth links for training
    └── test/
        ├── source1.csv        # Test Source 1 records  (1,732,544 entities)
        ├── source2.csv        # Test Source 2 records
        └── source3.csv        # Test Source 3 records
```

All commands below should be run from the **repository root** (`ml2/`).

---

## 5. End-to-End Reproduction Instructions

### Step 1 — Train the Unified 5-Fold LightGBM Ensemble

Trains across the combined India + US corpus (≈2.2M entities, ≈24.9M candidate pairs) using 5-fold cross-validation. Saves fold models to `models/v3/lgbm_fold*.txt` and calibrated thresholds to `decision.json`.

```bash
python code/business_entity_resolution/src/train_universe.py
```

**Expected output (approx.):**
```
[universe] OOF Macro F0.5 = 0.9087   Micro Precision = 96.84%
[universe] Saved 5 fold models → models/v3/
[universe] Saved decision thresholds → decision.json
```

> ⚠️ This step requires ~20–40 minutes on a modern CPU. Pre-trained models are included
> in `models/v3/` so you may skip to Step 2 if reproducing inference only.

### Step 2 — Run Full-Universe Inference (Best Submission)

Runs inference across all three countries (India, US, France) applying the unified ensemble with per-country conservative threshold offsets. Writes both output TSVs.

```bash
python code/business_entity_resolution/src/predict_universe.py
```

This generates:
- `output/matching_results_unified_unseen05.tsv` — final predicted matches
- `output/candidate_pairs_unified_unseen05.tsv` — blocking candidate set

**The files submitted to the leaderboard are the `_unified_unseen05` variants** (the `unseen05` suffix denotes the +0.05 threshold offset applied to the unseen France partition).

### Step 3 — Validate the Outputs

```bash
python datasource/utils/validate_submission.py \
    --matching output/matching_results_unified_unseen05.tsv \
    --candidate output/candidate_pairs_unified_unseen05.tsv \
    --test-dir datasource/dataset/test
```

Expected result: `PASS — no blocking issues found. Safe to submit.`

---

## 6. Pre-trained Models

The trained fold models are included in the repository under `models/v3/`. You can skip Step 1 and run inference directly with Step 2 using these pre-trained weights.

| File | Description |
|:---|:---|
| `models/v3/lgbm_fold0.txt` | Fold 0 — 300-tree LightGBM (India+US train) |
| `models/v3/lgbm_fold1.txt` | Fold 1 |
| `models/v3/lgbm_fold2.txt` | Fold 2 |
| `models/v3/lgbm_fold3.txt` | Fold 3 |
| `models/v3/lgbm_fold4.txt` | Fold 4 |
| `decision.json` | Calibrated per-source decision thresholds |

---

## 7. Key Design Decisions

| Decision | Rationale |
|:---|:---|
| **Unified India+US training** | US was zero-shot in earlier runs. Joining corpora added 1.4M US entities to training and raised leaderboard from 0.776 → 0.880. |
| **Contiguous int32 indexing** | Python object pointers for 6.2M US entities caused MemoryErrors. Switching to contiguous NumPy int32 arrays reduced peak RAM from OOM to ~7.8 GB. |
| **+0.05 France threshold offset** | France was entirely absent from training. A conservative precision-biased offset suppresses false merges on the unseen distribution. |
| **Global greedy bipartite exclusivity** | Enforces the competition constraint that each candidate in S2/S3 is matched to at most one S1 entity, eliminating guaranteed false positives. |
| **Devanagari → Latin transliteration** | Recovers India pairs where S1 is in English Latin and S2/S3 is in Hindi Devanagari script. Without transliteration, string similarity is near-zero. |

---

## 8. Submission Artifacts

| File | Description |
|:---|:---|
| `output/matching_results.tsv` | Final matches — identical to leaderboard upload |
| `output/candidate_pairs.tsv` | Blocking candidate set used for matching |

Both files comply with the official format: tab-separated, `source1_id` in column 1, space-separated `matched_ids` in column 2, one row per S1 entity.
