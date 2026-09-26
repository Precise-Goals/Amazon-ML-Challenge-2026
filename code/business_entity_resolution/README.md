# Business Entity Resolution Pipeline

High-performance, machine learning-driven Business Entity Resolution pipeline engineered for large-scale commercial entity deduplication and cross-source record linkage across millions of noisy records.

---

## 1. Overview & Architecture

Given business records across three independent sources (`Source 1`, `Source 2`, and `Source 3`):
- **Source 1** is the deduplicated reference source.
- The pipeline discovers all matching entities from **Source 2** and **Source 3** for each **Source 1** entity.
- The system is evaluated using the macro-averaged **$F_{0.5}$ score**, which weights precision $2\times$ over recall to heavily penalize false merges while rewarding correct singleton resolution.

### Pipeline Architecture:
```
  [ Source 1 Records ]        [ Source 2 Records ]        [ Source 3 Records ]
          │                           │                           │
          ▼                           └─────────────┬─────────────┘
  Country Partitioning                              ▼
          │                               Country Partitioning
          │                                         │
          │                                         ▼
          │                               Multi-Pass Inverted Index
          │                                (Token, Bigram, Prefix4,
          │                                  Address Zip & Numbers)
          │                                         │
          └───────────────────┬─────────────────────┘
                              ▼
                   Candidate Generation & 
                 Token-Set Pruning (Top-15)
                              │
                              ▼
                  Pair Feature Extraction
                 (RapidFuzz Name & Address
                  Sims, Numbers, Geocodes)
                              │
                              ▼
                   LightGBM GBDT Classifier
                   (Binary Cross-Entropy)
                              │
                              ▼
                Macro F_0.5 Threshold Tuning
                      (Threshold: ~0.60)
                              │
                              ▼
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
  matching_results.tsv                  candidate_pairs.tsv
(Final Predicted Matches)              (Blocking Candidates)
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Module definitions
│   ├── config.py             # Hyperparameters, filepaths, and stopping rules
│   ├── utils.py              # Text normalizers, address parser, macro F_0.5 scorer
│   ├── blocking.py           # Multi-pass inverted index blocking engine
│   ├── features.py           # C++-accelerated RapidFuzz feature extraction
│   ├── model.py              # LightGBM classifier wrapper & threshold tuner
│   └── pipeline.py           # End-to-end command-line runner
├── models/
│   └── lgbm_model.joblib     # Persisted trained model & threshold
├── README.md                 # Reproduction guide & documentation
└── requirements.txt          # Pinned Python package dependencies
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.9+ (Python 3.10 - 3.14 supported)
- 8GB+ RAM recommended

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction Instructions

### Option A: Complete Pipeline (Train + Optimize + Predict)
To execute model training on the training corpus, calibrate the decision threshold for macro $F_{0.5}$, and generate both submission files for the test dataset:
```bash
python code/business_entity_resolution/src/pipeline.py --mode all
```

### Option B: Model Training Only
```bash
python code/business_entity_resolution/src/pipeline.py --mode train --sample-size 100000
```

### Option C: Inference on Test Set Only
To run inference using the pre-trained model saved in `models/lgbm_model.joblib`:
```bash
python code/business_entity_resolution/src/pipeline.py --mode predict
```

---

## 5. Submission Artifacts & Verification

The pipeline writes the final submission files directly to `output/`:
- `output/matching_results.tsv`: Final predictions formatted as `source1_entity_id \t matched_entity_ids`.
- `output/candidate_pairs.tsv`: Final blocking candidate set fed into the model.

### Format Validation
Run the official competition validator:
```bash
python datasource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir datasource/dataset/test
```
A return code of `0` and `PASS` confirms that all constraints, headers, formatting, deduplication, and candidate subset rules are satisfied.
