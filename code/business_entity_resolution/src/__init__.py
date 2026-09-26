"""
Business Entity Resolution System Package.
"""

from .config import config
from .utils import normalize_text, calculate_macro_f05
from .blocking import BlockingEngine
from .features import extract_pair_features, FEATURE_NAMES
from .model import EntityResolutionModel

__all__ = [
    'config',
    'normalize_text',
    'calculate_macro_f05',
    'BlockingEngine',
    'extract_pair_features',
    'FEATURE_NAMES',
    'EntityResolutionModel'
]
