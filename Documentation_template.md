# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER Lab  
**Team Members:** Machine Learning Engineering Team  
**Submission Date:** 2026-09-26  
**Latest Official Portal Score:** **`0.776`** (Macro $F_{0.5}$)  

---

## 1. Executive Summary
We present a scalable, high-precision Business Entity Resolution system engineered using an advanced **Medallion Data Architecture (Bronze $\rightarrow$ Silver $\rightarrow$ Gold)** to link fragmented and noisy commercial entity records across multiple disparate sources (Source 1, Source 2, and Source 3) without external data lookup. Our approach couples a zero-copy streaming Bronze ingestion layer with a Silver canonicalization layer (featuring phonetic Devanagari transliteration, French & English legal form isolation, and address normalization) and a Gold precision resolution engine. The Gold engine features multi-pass frequency-capped blocking, a **43-feature Gradient Boosted Decision Tree matcher** with **Group Context** and **Reverse Context**, **Isotonic Probability Calibration**, an **Asymmetric Decision Boundary** ($\theta_{S_2} = 0.70, \theta_{S_3} = 0.65$), a **Top-12 Learned Candidate Pruner**, and a **Global Greedy Bipartite Exclusivity Layer**. On the full 1.73M entity test set, the pipeline achieves **100% compliance** with the official competition validator, resolving **15,255 contested candidate collisions**, achieving **3.76 average matches per entity** (closely matching the ground-truth benchmark of 3.66), and reaching a new peak portal score of **`0.776`**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across over 12 million business records revealed critical noise profiles and structural properties:
1. **Strict Country Isolation:** Across all 7,638,365 ground-truth pairs in the training corpus, 100% of matches occur strictly within the same country ($S_1.\text{country} == S_{2/3}.\text{country}$). There are zero cross-country entity matches. This enables zero-loss partitioning by country (US, India, France).
2. **Multilingual Transliteration (Devanagari/Hindi):** In the Indian subset, thousands of records represent identical businesses where Source 1 uses the English Latin name while Source 2/3 uses Hindi Devanagari script (e.g., `Raj Best Investment` vs `र ज बसट इनवसटमट`, `Sunrise Solutions` vs `सनर इज स लयशस`). Raw string metrics gave near-zero similarity ($< 0.10$). Phonetic Devanagari transliteration unifies cross-script pairs and recovers missed matches.
3. **Over-Prediction & Macro $F_{0.5}$ Asymmetry:** The evaluation metric is Macro $F_{0.5}$, where precision is weighted $2\times$ over recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$). A single false merge penalizes the entity score drastically. Singletons (entities with zero true matches, ~5.6% of ground truth entities) earn a full 1.0 when left empty, but drop to 0.0 on any false positive merge.
4. **The 1-to-N Candidate Duplication Invariant:** In the ground truth, every physical business candidate in $S_2$ or $S_3$ belongs to at most **one** $S_1$ entity. Assigning a candidate to multiple $S_1$ entities injects guaranteed false positive links.
5. **Commercial Co-Tenant & Chain Confusion:** Unrelated businesses sharing a commercial plaza/building address or chain store names with conflicting street numbers cause high-similarity false positives. Enforcing group context and street number consistency suppresses these distractors.

### 2.2 Solution Strategy: Medallion Architecture
We implemented a structured Medallion Data Architecture:
- **Bronze Layer (Ingestion & Partitioning):** Vectorized DuckDB streaming partition by country (`India`, `US`, `France`), eliminating memory leaks and enforcing zero-copy reads.
- **Silver Layer (Canonicalization & Normalization):**
  - Unicode accent folding (`NFD` normalization).
  - URL prefixes, handles, and domain stripping (`.com`, `.net`, `.in`, `.fr`).
  - Phonetic transliteration of Devanagari Hindi characters to Latin phonetics.
  - Corporate legal entity form extraction (`pvtltd`, `publicltd`, `llc`, `inc`, `corp`, `sarl`, `sa`, `sci`, `sasu`, `eurl`).
  - Street abbreviation expansion (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ste` $\rightarrow$ `suite`, `bd` $\rightarrow$ `boulevard`, `av` $\rightarrow$ `avenue`).
  - Discrete numeric anchor & postal code isolation.
- **Gold Layer (High-Precision Matching Engine):**
  - Multi-pass inverted index blocking with posting frequency capping.
  - Composite candidate ranking combining brand and address saliency.
  - **43 dense interaction, group-context, and reverse-context features**.
  - Isotonic probability calibration fitted on out-of-fold cross-validation.
  - Asymmetric per-source decision boundaries ($\theta_{S_2} = 0.70, \theta_{S_3} = 0.65$).
  - Global greedy bipartite exclusivity on contested candidates.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73\text{M} \times 10\text{M} \approx 17.3 \text{ Trillion}$ comparison space to a scalable candidate set, we implement a multi-pass inverted index:

- **Blocking keys used:**
  1. `('tok', country, token)`: Non-stop name unigrams with length $\ge 3$.
  2. `('bi', country, token1_token2)`: Adjacent name token bigrams for multi-word phrases.
  3. `('pref4', country, prefix4)`: 4-character prefix of the primary name token.
  4. `('addr_tok', country, word)`: Informative address tokens (length $\ge 4$) filtered by address stopwords.
  5. `('addr_pair', country, word1, word2)`: Distinct pairs of informative address tokens.
  6. `('zip_num', country, zip_code, street_num)`: Postal code combined with street number.
  7. `('sn_w', country, street_num, street_word)`: Street number combined with street name token.
  8. `('zip_w', country, zip_code, street_word)`: Postal code combined with primary street token.
  9. `('st_zip', country, short_tok, zip_code)` & `('st_sn', country, short_tok, street_num)`: 2-character name acronyms paired with address numbers.
- **Candidate pairs generated:**
  - Raw blocking achieves **96.76% candidate recall** across all countries.
  - Learned candidate pruner caps candidates at **Top-12** most plausible matches, retaining **99.4% reachable recall** while reducing candidate pool size to **11.14 candidates per Source 1 entity** on the 1.73M test set (273 MB archive vs 328 MB in baseline).

---

## 4. Matching Model

**Features used (43 dense features):**
- **Pairwise Name Metrics (5):** `n_ratio`, `n_partial`, `n_tsort`, `n_tset`, `n_wratio`.
- **Pairwise Address Metrics (4):** `a_ratio`, `a_partial`, `a_tsort`, `a_tset`.
- **Interaction Metrics (3):** `max_sim`, `comb_sim` ($0.6 \cdot \text{name} + 0.4 \cdot \text{addr}$), `dual_high`.
- **Discrete & Geospatial Anchors (7):** `num_overlap`, `num_match`, `num_conflict`, `zip_match`, `zip_conflict`, `addr_missing`, `strict_num_conflict`, `num_missing_one_side`.
- **Corporate Legal Structure (2):** `legal_match`, `legal_conflict`.
- **Discriminative Identity (3):** `name_exact`, `brand_len`, `word_cnt`, `has_translit`.
- **Group Context Features (12):** `rank_comb`, `rank_name`, `gap_comb`, `gap_name`, `z_comb`, `n_cands`, `n_s2`, `n_s3`, `rank_within_src`, `agree_max_other`, `agree_mean_other`, `count_sim_gt_90`.
- **Reverse Context Features (3):** `cand_s1_count`, `is_mutual_best`, `gap_to_second_s1`.
- **Missing Address Flags (2):** `s1_addr_missing`, `c_addr_missing`.

**Model Type & Calibration:**
- **LightGBM (Gradient Boosted Decision Trees)** binary classifier with 300 estimators, max depth 6, 31 leaves.
- **Isotonic Calibrator:** Maps raw margin outputs into true empirical posterior probabilities.
- **Asymmetric Decision Layer:** $\theta_{S_2} = 0.70, \theta_{S_3} = 0.65$.
- **Global Greedy Bipartite Exclusivity:** Contested candidate IDs claimed by multiple $S_1$ entities are awarded strictly to the highest-probability $S_1$ claimant, guaranteeing zero duplicate candidate assignments.

---

## 5. Results & Error Analysis

- **Official Leaderboard Score:** **`0.776`** (Macro $F_{0.5}$).
- **Hold-out Validation Macro $F_{0.5}$:** **`0.9643`**.
- **Average Matches per Entity:** **`3.76`** (vs Ground Truth benchmark of `3.66`).
- **Singletons Rate:** **`4.76%`** (`82,438` entities), restoring singleton balance towards the ground-truth benchmark (~5.58%).
- **Candidate Set Size:** **`11.14`** candidates per Source 1 entity (Top-12 capped).
- **Validation Compliance:** Checked with official `datasource/utils/validate_submission.py` $\rightarrow$ **PASS: 100% compliant**.

### Country Partition Diagnostics

| Partition | Entities | Avg Matches | Singletons | Key Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **India** | 809,986 | **3.69** | **6.44%** | Near-perfect alignment with Ground Truth (3.66). |
| **US** | 663,106 | **3.99** | **2.22%** | Franchise/chain overmatching suppressed singletons. |
| **France** | 259,452 | **3.36** | **5.99%** | Zero-shot transfer (France was absent from training). |

---

## 6. Conclusion
The advanced Medallion Data Architecture delivers an end-to-end, memory-safe, reproducible Business Entity Resolution pipeline that unifies streaming ingestion, multilingual canonicalization, 43-feature context-aware tree matching, probability calibration, and greedy bipartite exclusivity. It achieves an official leaderboard score of **`0.776`**, executing across 1.73M test entities and 10M candidate pool records with complete memory stability and zero external data dependencies.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py               # Package interface
│   ├── config.py                 # Hyperparameters and path configurations
│   ├── bronze.py                 # Bronze Layer: zero-copy ingestion & country partition
│   ├── silver.py                 # Silver Layer: normalization, Devanagari transliteration, legal forms
│   ├── gold.py                   # Gold Layer: multi-pass blocking, dense features, precision gates
│   ├── features_v2.py            # Group context, reverse context, and agreement features
│   ├── predict_submission.py     # Production inference runner with disk-backed streaming
│   ├── train_phase_b.py          # Phase B 43-feature LightGBM model trainer
│   ├── phase_c_decision.py       # Phase C decision boundary optimizer
│   ├── phase_d_e_blocking.py     # Phase D/E blocking & candidate pruner
│   └── utils.py                  # Text normalization, address parsing, macro F_0.5 metric
├── models/
│   ├── lgbm_matcher_v2.joblib    # Trained 43-feature LightGBM matcher
│   ├── isotonic_calibrator.joblib# Trained Isotonic probability calibrator
│   └── phase_b_meta.json         # Feature list and metadata
├── README.md                     # Step-by-step reproduction instructions
└── requirements.txt              # Pinned dependency environment
```

**Reproduction Command:**
```bash
python code/business_entity_resolution/src/predict_submission.py
```

### B. Official Validation
The generated submission files have been validated with `datasource/utils/validate_submission.py`:
```
ML Challenge 2026 — submission validator
  test dir: D:\Workspace\Projects\ml2\datasource\dataset\test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (82438 empty, 1650106 non-empty).
  candidate_pairs.tsv: 1732544 rows (23559 empty, 1708985 non-empty).

PASS — no blocking issues found. Safe to submit.
```
