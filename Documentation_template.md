# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER Lab  
**Team Members:** Machine Learning Engineering Team  
**Submission Date:** 2026-09-26  

---

## 1. Executive Summary
We present a scalable, high-precision Business Entity Resolution system engineered using a **Medallion Data Architecture (Bronze $\rightarrow$ Silver $\rightarrow$ Gold)** to link fragmented and noisy commercial entity records across multiple disparate sources (Source 1, Source 2, and Source 3) without external data lookup. Our approach couples a zero-copy streaming Bronze ingestion layer with a Silver canonicalization layer (featuring phonetic Devanagari transliteration, legal form isolation, and address normalization) and a Gold precision resolution engine (featuring multi-pass frequency-capped blocking, 24 dense features, and deterministic precision verification quality gates). On hold-out validation splits, the pipeline achieves **98.30% - 98.93% precision**, **92.22% recall**, and an optimal macro $F_{0.5}$ score of **0.9581**, while maintaining a compact candidate pool averaging **13.63 candidates per Source 1 entity** on the 1.73M test set with **4.61% singletons** closely mirroring the ground truth baseline.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across over 12 million business records revealed critical noise profiles and structural properties:
1. **Strict Country Isolation:** Across all 7,638,365 ground-truth pairs in the training corpus, 100% of matches occur strictly within the same country ($S_1.\text{country} == S_{2/3}.\text{country}$). There are zero cross-country entity matches. This enables zero-loss partitioning by country (US, India, France).
2. **Multilingual Transliteration (Devanagari/Hindi):** In the Indian subset, thousands of records represent identical businesses where Source 1 uses the English Latin name while Source 2/3 uses Hindi Devanagari script (e.g., `Raj Best Investment` vs `र ज बसट इनवसटमट`, `Sunrise Solutions` vs `सनर इज स लयशस`). Raw string metrics gave near-zero similarity ($< 0.10$). Phonetic Devanagari transliteration unifies cross-script pairs and recovers missed matches.
3. **Over-Prediction & Macro $F_{0.5}$ Asymmetry:** The evaluation metric is Macro $F_{0.5}$, where precision is weighted $2\times$ over recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$). A single false merge penalizes the entity score drastically. Singletons (entities with zero true matches, ~5.6% of ground truth entities) earn a full 1.0 when left empty, but drop to 0.0 on any false positive merge.
4. **Distractor Suppression:** Unrelated businesses sharing a commercial plaza/building address or chain store names with conflicting street numbers/postal codes cause high-similarity false positives. Enforcing deterministic precision gates suppresses these distractors and preserves singletons.

### 2.2 Solution Strategy: Medallion Architecture
We implemented a structured Medallion Data Architecture:
- **Bronze Layer (Ingestion & Partitioning):** Vectorized DuckDB streaming partition by country (`India`, `US`, `France`), eliminating memory leaks and enforcing zero-copy reads.
- **Silver Layer (Canonicalization & Normalization):**
  - Unicode accent folding (`NFD` normalization).
  - URL prefixes, handles, and domain stripping (`.com`, `.net`, `.in`, `.fr`).
  - Phonetic transliteration of Devanagari Hindi characters to Latin phonetics.
  - Corporate legal entity form extraction (`pvtltd`, `publicltd`, `llc`, `inc`, `corp`, `sarl`, `sa`, `sci`).
  - Street abbreviation expansion (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ste` $\rightarrow$ `suite`).
  - Discrete numeric anchor & postal code isolation.
- **Gold Layer (High-Precision Matching Engine):**
  - Multi-pass inverted index blocking with posting frequency capping.
  - Composite candidate ranking combining brand and address saliency.
  - 24 dense interaction features.
  - Multi-tier precision verification quality gates for hard negative conflict suppression.

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
  - Dual-similarity ranking prunes candidates to the Top-15 most plausible matches, achieving an average of **13.63 candidates per Source 1 entity** across the entire 1.73M test set.

---

## 4. Matching Model

**Features used (24 dense features):**
- **Name Metrics:** `n_ratio`, `n_partial`, `n_tsort`, `n_tset`, `n_wratio`.
- **Address Metrics:** `a_ratio`, `a_partial`, `a_tsort`, `a_tset`.
- **Interaction Metrics:** `max_sim`, `comb_sim` ($0.6 \cdot \text{name} + 0.4 \cdot \text{addr}$), `dual_high`.
- **Discrete & Geospatial:** `num_overlap`, `num_match`, `num_conflict`, `zip_match`, `zip_conflict`, `addr_missing`.
- **Corporate Legal Structure:** `legal_match`, `legal_conflict`.
- **Discriminative Identity:** `name_exact`, `brand_len`, `word_cnt`, `has_translit`.

**Model Type:**
- **LightGBM (Gradient Boosted Decision Trees)** binary classifier with 300 estimators, max depth of 6, 31 leaves, subsample 0.8, and feature subsampling 0.8.
- Inference rate: $> 120{,}000$ pairs/second per CPU core.

**Medallion Gold Precision Verification Quality Gates:**
1. *Gate 1 (Numeric Conflict):* Reject if records have conflicting street numbers, low address similarity, and non-identical names.
2. *Gate 2 (Postal Conflict):* Reject if postal codes conflict and address similarity is low.
3. *Gate 3 (Legal Form Conflict):* Reject corporate legal mismatches (e.g. `public limited` vs `private limited`) unless high address similarity confirms identity.
4. *Gate 4 (Missing Address Gating):* When address is missing in either record, require high brand name confidence ($\ge 0.80$).
5. *Gate 5 (Plaza Distractor Suppression):* Reject unrelated businesses sharing a commercial plaza/strip mall address when names are disjoint ($\text{name\_tset} < 0.35$).
6. *Gate 6 (Calibrated ML Threshold):* Active decision threshold $\theta = 0.75$, optimized via grid search on macro $F_{0.5}$.

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (macro):** **0.9581** on hold-out validation split.
- **Global Precision:** **98.30% - 98.93%**; **Global Recall:** **92.22%**.
- **Candidate Set Size:** **13.63** candidates per Source 1 entity.
- **Singletons Rate:** **4.61%** singletons on test set (79,956 entities), closely matching the ground truth baseline (~5.5%).
- **Average Matches per Entity:** **4.03** matches per non-empty entity.
- **Validation Compliance:** Checked with official `datasource/utils/validate_submission.py` $\rightarrow$ **PASS: 100% compliant**.

---

## 6. Conclusion
The Medallion Data Architecture delivers an end-to-end, reproducible, highly scalable Business Entity Resolution pipeline that unifies streaming ingestion (Bronze), multilingual canonicalization (Silver), and high-precision matching (Gold). It achieves an ultra-compact candidate set averaging **13.63 candidates per Source 1 entity** with **98.3%+ precision** and an $F_{0.5}$ score of **0.9581**, executing across 1.73M test entities and 10M candidate pool records with zero external data dependencies.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py         # Package interface
│   ├── config.py           # Hyperparameters and path configurations
│   ├── bronze.py           # Bronze Layer: zero-copy ingestion & country partition
│   ├── silver.py           # Silver Layer: normalization, Devanagari transliteration, legal forms
│   ├── gold.py             # Gold Layer: multi-pass blocking, 24 dense features, precision gates
│   ├── model.py            # LightGBM classifier and F_0.5 threshold tuner
│   ├── utils.py            # Text normalization, address parsing, macro F_0.5 metric
│   ├── blocking.py         # Inverted index blocking engine
│   ├── features.py         # Pairwise RapidFuzz feature extraction
│   └── pipeline.py         # End-to-end execution runner (train, predict, all)
├── models/
│   └── lgbm_model.joblib   # Serialized trained model & threshold
├── README.md               # Step-by-step reproduction instructions
└── requirements.txt        # Pinned dependency environment
```

**Reproduction Command:**
```bash
python code/business_entity_resolution/src/pipeline.py --mode all
```

### B. Official Validation
The generated submission files (`output/matching_results.tsv` and `output/candidate_pairs.tsv`) have been validated with `datasource/utils/validate_submission.py`:
```
PASS — no blocking issues found. Safe to submit.
```
