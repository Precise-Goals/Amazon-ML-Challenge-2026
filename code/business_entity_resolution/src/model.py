"""
Model training, inference, and F_0.5 threshold optimization.
"""

import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Set, Tuple
try:
    from .config import config
    from .utils import calculate_macro_f05
except (ImportError, ValueError):
    from config import config
    from utils import calculate_macro_f05


class EntityResolutionModel:
    def __init__(self, params: dict = None, threshold: float = None):
        self.params = params if params is not None else config.LGBM_PARAMS
        self.threshold = threshold if threshold is not None else config.DEFAULT_THRESHOLD
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM binary classifier on feature matrix."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for candidate pairs."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        pair_keys: List[Tuple[str, str]],
        val_s1_ids: Set[str],
        ground_truth: Dict[str, Set[str]],
        thresholds: List[float] = None
    ) -> Tuple[float, float]:
        """
        Grid search for decision threshold that maximizes Macro F_0.5 on validation split.
        """
        if thresholds is None:
            thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            
        probs = self.predict_proba(X_val)
        best_th = self.threshold
        best_score = -1.0
        
        val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_ids}
        for th in thresholds:
            th = round(float(th), 2)
            preds_by_s1 = {s1_id: [] for s1_id in val_s1_ids}
            for (s1_id, cid), prob in zip(pair_keys, probs):
                if prob >= th:
                    preds_by_s1[s1_id].append(cid)
                    
            f05 = calculate_macro_f05(val_gt, preds_by_s1)
            empty_cnt = sum(1 for sid, m in preds_by_s1.items() if len(m) == 0)
            print(f"  Threshold {th:.2f}: Macro F_0.5 = {f05:.4f} (Singletons: {empty_cnt/len(val_s1_ids)*100:.2f}%)")

            if f05 > best_score:
                best_score = f05
                best_th = th
                
        self.threshold = best_th
        return best_th, best_score

    def save(self, filepath: str):
        """Persist model and optimal threshold to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.threshold}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EntityResolutionModel":
        """Load trained model and threshold from disk."""
        data = joblib.load(filepath)
        instance = cls(threshold=data.get('threshold', config.DEFAULT_THRESHOLD))
        instance.model = data['model']
        instance.is_fitted = True
        return instance
