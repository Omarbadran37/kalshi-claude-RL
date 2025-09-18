"""
NFL Trading Backtesting Infrastructure

This module provides comprehensive backtesting capabilities for NFL trading strategies,
including simulation environments, baseline strategies, and performance analytics.
"""

from .trading_environment import TradingEnvironment
from .strategies import (
    BaseStrategy,
    RuleBasedTrader,
    StatisticalTrader,
    RandomTrader,
    BuyAndHoldTrader
)
from .backtester import Backtester
from .performance_analyzer import PerformanceAnalyzer
from .reporting import ReportGenerator

__all__ = [
    'TradingEnvironment',
    'BaseStrategy',
    'RuleBasedTrader',
    'StatisticalTrader', 
    'RandomTrader',
    'BuyAndHoldTrader',
    'Backtester',
    'PerformanceAnalyzer',
    'ReportGenerator'
]