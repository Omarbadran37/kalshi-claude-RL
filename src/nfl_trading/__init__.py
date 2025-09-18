"""NFL Momentum Trading System

A reinforcement learning system for predicting NFL price movements
on Kalshi prediction markets using play-by-play data.
"""

__version__ = "0.1.0"
__author__ = "NFL Trading Team"

from .data import NFLDataProcessor, KalshiDataProcessor, DataAligner
from .config import Config, LoggerConfig

__all__ = [
    "NFLDataProcessor",
    "KalshiDataProcessor", 
    "DataAligner",
    "Config",
    "LoggerConfig"
]