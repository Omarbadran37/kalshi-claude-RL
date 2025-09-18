"""
NFL Trading ML Inference Module

Real-time prediction and inference capabilities.
"""

from .play_predictor import PlayPredictor, PredictionResult

__all__ = ['PlayPredictor', 'PredictionResult']