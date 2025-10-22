"""
Package de classification multi-label pour les tags de problèmes de programmation
"""

from .data import DataProcessor
from .models import TFIDFModel, HybridModel
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__all__ = ["DataProcessor", "TFIDFModel", "HybridModel", "ModelEvaluator"]