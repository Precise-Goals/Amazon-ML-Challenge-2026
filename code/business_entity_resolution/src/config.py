"""
Configuration and Hyperparameters for Business Entity Resolution Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Set

@dataclass
class Config:
    # Directory paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATASET_DIR: str = os.path.join(BASE_DIR, "datasource", "dataset")
    TRAIN_DIR: str = os.path.join(DATASET_DIR, "train")
    TEST_DIR: str = os.path.join(DATASET_DIR, "test")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "..", "models")
    
    # Input files
    TRAIN_S1: str = os.path.join(TRAIN_DIR, "train_source1.tsv")
    TRAIN_S2: str = os.path.join(TRAIN_DIR, "train_source2.tsv")
    TRAIN_S3: str = os.path.join(TRAIN_DIR, "train_source3.tsv")
    TRAIN_GT: str = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    
    TEST_S1: str = os.path.join(TEST_DIR, "test_source1.tsv")
    TEST_S2: str = os.path.join(TEST_DIR, "test_source2.tsv")
    TEST_S3: str = os.path.join(TEST_DIR, "test_source3.tsv")
    
    # Output files
    MATCHING_OUTPUT: str = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_OUTPUT: str = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    # Blocking parameters
    MAX_POSTINGS_PER_KEY: int = 50       # Max candidate matches per individual key
    TOP_K_CANDIDATES: int = 15           # Maximum candidates retained per S1 entity
    UNIVERSE_TOP_K: int = 12             # v3: one top_k shared by train and test universes
    MIN_TOKEN_LEN: int = 3               # Minimum character length for tokens
    PREFIX_LEN: int = 4                  # Character prefix length
    
    # Common stop-words and legal entity designators to filter from keys
    NAME_STOP_WORDS: Set[str] = field(default_factory=lambda: {
        'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
        'pvt', 'private', 'co', 'company', 'sa', 'sarl', 'gmbh', 'services', 'solutions',
        'enterprises', 'traders', 'group', 'india', 'hotel', 'shree', 'sri', 'new',
        'dr', 'mr', 'mrs', 'saint', 'restaurant', 'cafe', 'bar', 'store', 'shop',
        'sas', 'sasu', 'eurl', 'sci', 'snc', 'france', 'association', 'societe', 'club', 'centre'
    })
    
    ADDR_STOP_WORDS: Set[str] = field(default_factory=lambda: {
        'near', 'behind', 'opp', 'opposite', 'shop', 'no', 'floor', 'suite', 'ste',
        'unit', 'apt', 'apartment', 'bldg', 'building', 'road', 'rd', 'street', 'st',
        'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'court', 'ct', 'boulevard', 'blvd', 'bd',
        'highway', 'hwy', 'phase', 'block', 'sector', 'post', 'po', 'box', 'pmb',
        'nagar', 'colony', 'marg', 'puram', 'pradesh', 'state', 'district', 'dist',
        'bengal', 'maharashtra', 'karnataka', 'tamil', 'nadu', 'delhi', 'mumbai',
        'calcutta', 'kolkata', 'chennai', 'hyderabad', 'bangalore', 'bengaluru',
        'village', 'post', 'taluk', 'tehsil', 'west', 'east', 'north', 'south',
        'rue', 'chemin', 'impasse', 'cours', 'allee', 'place', 'route', 'faubourg',
        'de', 'du', 'la', 'des', 'le', 'les'
    })
    
    # LightGBM Classifier Parameters
    LGBM_PARAMS: dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 4,
        'verbose': -1
    })
    
    # Classification threshold for F_0.5 score
    DEFAULT_THRESHOLD: float = 0.70
    
    # Training sample size for model fitting (representative subset)
    TRAIN_SAMPLE_SIZE: int = 100000

config = Config()
