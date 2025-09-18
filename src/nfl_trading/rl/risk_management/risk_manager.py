"""
Risk Management System

Advanced risk management for RL trading agents including Kelly criterion,
dynamic stop-loss, portfolio heat limits, and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring"""
    portfolio_value: float
    total_exposure: float
    max_individual_position: float
    correlation_risk: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk


@dataclass
class PositionLimits:
    """Position size limits and constraints"""
    max_portfolio_heat: float = 0.02  # 2% portfolio heat limit
    max_position_size: int = 500
    max_correlation_exposure: float = 0.8  # Max 80% in correlated positions
    max_sector_exposure: float = 0.6  # Max 60% in single sector
    min_trade_size: int = 10
    max_leverage: float = 1.0  # No leverage by default


@dataclass
class StopLossConfig:
    """Stop-loss configuration"""
    static_stop_pct: float = 0.05  # 5% static stop-loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    dynamic_stop_enabled: bool = True
    volatility_multiplier: float = 2.0  # Stop = volatility * multiplier
    time_based_stops: bool = True
    max_hold_time_minutes: int = 60  # Maximum position hold time


class RegimeDetector:
    """Market regime detection for dynamic risk adjustment"""

    def __init__(self, lookback_window: int = 100, n_regimes: int = 3):
        self.lookback_window = lookback_window
        self.n_regimes = n_regimes
        self.price_history = deque(maxlen=lookback_window)
        self.volume_history = deque(maxlen=lookback_window)
        self.regime_model = None
        self.current_regime = 0
        self.regime_labels = ['Low Volatility', 'Normal', 'High Volatility']

    def update(self, price: float, volume: float):
        """Update regime detection with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) >= 20:  # Minimum data for regime detection
            self._detect_regime()

    def _detect_regime(self):
        """Detect current market regime using price and volume features"""
        if len(self.price_history) < 20:
            return

        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))

        # Calculate features for regime detection
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        volume_ratio = volumes[-1] / np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0
        price_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0

        # Simple regime classification based on volatility
        if volatility < 0.01:
            self.current_regime = 0  # Low volatility
        elif volatility > 0.03:
            self.current_regime = 2  # High volatility
        else:
            self.current_regime = 1  # Normal

    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime information"""
        return {
            'regime_id': self.current_regime,
            'regime_name': self.regime_labels[self.current_regime],
            'confidence': 0.8,  # Simplified confidence
            'features': {
                'volatility': np.std(np.diff(list(self.price_history))) if len(self.price_history) > 1 else 0.01,
                'trend': (self.price_history[-1] - self.price_history[0]) / self.price_history[0] if len(self.price_history) > 1 else 0.0
            }
        }

    def get_risk_multiplier(self) -> float:
        """Get risk multiplier based on current regime"""
        multipliers = [0.8, 1.0, 1.5]  # Low, Normal, High volatility
        return multipliers[self.current_regime]


class KellyCriterion:
    """Kelly criterion for optimal position sizing"""

    def __init__(self, lookback_window: int = 100, min_trades: int = 20):
        self.lookback_window = lookback_window
        self.min_trades = min_trades
        self.trade_history = deque(maxlen=lookback_window)

    def add_trade_result(self, pnl: float, position_size: int, price: float):
        """Add trade result for Kelly calculation"""
        if position_size != 0 and price > 0:
            trade_return = pnl / (abs(position_size) * price)
            self.trade_history.append(trade_return)

    def calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly fraction for optimal position sizing.

        Kelly fraction = (bp - q) / b
        where:
        - b = odds (average win / average loss)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        if len(self.trade_history) < self.min_trades:
            return 0.1  # Conservative default

        returns = np.array(list(self.trade_history))

        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.1  # Conservative if no wins or losses

        # Calculate Kelly parameters
        win_prob = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = np.mean(np.abs(losses))

        # Kelly fraction calculation
        if avg_loss > 0:
            odds_ratio = avg_win / avg_loss
            kelly_fraction = (win_prob * odds_ratio - (1 - win_prob)) / odds_ratio
        else:
            kelly_fraction = 0.1

        # Apply safety constraints
        kelly_fraction = np.clip(kelly_fraction, 0.01, 0.25)  # Min 1%, max 25%

        return kelly_fraction

    def get_optimal_position_size(
        self,
        portfolio_value: float,
        price: float,
        confidence: float = 1.0
    ) -> int:
        """Get optimal position size based on Kelly criterion"""
        kelly_fraction = self.calculate_kelly_fraction()

        # Adjust for confidence in trade
        adjusted_kelly = kelly_fraction * confidence

        # Calculate position size
        dollar_amount = portfolio_value * adjusted_kelly
        position_size = int(dollar_amount / price) if price > 0 else 0

        return position_size


class DynamicStopLoss:
    """Dynamic stop-loss management"""

    def __init__(self, config: StopLossConfig):
        self.config = config
        self.position_stops: Dict[str, Dict[str, Any]] = {}
        self.price_history: Dict[str, deque] = {}

    def set_position_stop(
        self,
        ticker: str,
        entry_price: float,
        position_size: int,
        entry_time: float
    ):
        """Set stop-loss for new position"""
        direction = 1 if position_size > 0 else -1

        # Calculate initial stop levels
        static_stop = entry_price * (1 - direction * self.config.static_stop_pct)

        # Dynamic stop based on recent volatility
        if ticker in self.price_history and len(self.price_history[ticker]) > 10:
            recent_prices = list(self.price_history[ticker])[-10:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            dynamic_stop_pct = volatility * self.config.volatility_multiplier
            dynamic_stop = entry_price * (1 - direction * dynamic_stop_pct)
        else:
            dynamic_stop = static_stop

        # Use the closer stop (more conservative)
        initial_stop = min(static_stop, dynamic_stop) if direction > 0 else max(static_stop, dynamic_stop)

        self.position_stops[ticker] = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'entry_time': entry_time,
            'current_stop': initial_stop,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'trailing_stop_active': False
        }

        logger.debug(f"Set stop for {ticker}: entry=${entry_price:.4f}, stop=${initial_stop:.4f}")

    def update_stops(self, ticker: str, current_price: float, current_time: float) -> Optional[str]:
        """
        Update stop-loss levels and check for triggered stops.

        Returns:
            Stop reason if position should be closed, None otherwise
        """
        if ticker not in self.position_stops:
            return None

        stop_info = self.position_stops[ticker]
        direction = stop_info['direction']

        # Update price history
        if ticker not in self.price_history:
            self.price_history[ticker] = deque(maxlen=100)
        self.price_history[ticker].append(current_price)

        # Update price extremes
        if direction > 0:  # Long position
            stop_info['highest_price'] = max(stop_info['highest_price'], current_price)

            # Check static stop
            if current_price <= stop_info['current_stop']:
                return "static_stop_loss"

            # Update trailing stop
            if self.config.trailing_stop_pct > 0:
                new_trailing_stop = stop_info['highest_price'] * (1 - self.config.trailing_stop_pct)
                if new_trailing_stop > stop_info['current_stop']:
                    stop_info['current_stop'] = new_trailing_stop
                    stop_info['trailing_stop_active'] = True

        else:  # Short position
            stop_info['lowest_price'] = min(stop_info['lowest_price'], current_price)

            # Check static stop
            if current_price >= stop_info['current_stop']:
                return "static_stop_loss"

            # Update trailing stop
            if self.config.trailing_stop_pct > 0:
                new_trailing_stop = stop_info['lowest_price'] * (1 + self.config.trailing_stop_pct)
                if new_trailing_stop < stop_info['current_stop']:
                    stop_info['current_stop'] = new_trailing_stop
                    stop_info['trailing_stop_active'] = True

        # Time-based stop
        if self.config.time_based_stops:
            hold_time_minutes = (current_time - stop_info['entry_time']) / 60
            if hold_time_minutes > self.config.max_hold_time_minutes:
                return "time_based_stop"

        return None

    def remove_position_stop(self, ticker: str):
        """Remove stop-loss tracking for closed position"""
        if ticker in self.position_stops:
            del self.position_stops[ticker]


class RiskManager:
    """
    Comprehensive risk management system for RL trading agents.

    Features:
    - Kelly criterion position sizing
    - Dynamic stop-loss management
    - Portfolio heat and correlation limits
    - Regime-based risk adjustment
    - Real-time risk monitoring
    """

    def __init__(
        self,
        position_limits: PositionLimits = None,
        stop_loss_config: StopLossConfig = None,
        enable_regime_detection: bool = True,
        enable_kelly_sizing: bool = True
    ):
        self.position_limits = position_limits or PositionLimits()
        self.stop_loss_config = stop_loss_config or StopLossConfig()
        self.enable_regime_detection = enable_regime_detection
        self.enable_kelly_sizing = enable_kelly_sizing

        # Initialize components
        self.regime_detector = RegimeDetector() if enable_regime_detection else None
        self.kelly_calculator = KellyCriterion() if enable_kelly_sizing else None
        self.stop_loss_manager = DynamicStopLoss(self.stop_loss_config)

        # Risk monitoring
        self.portfolio_history = deque(maxlen=1000)
        self.position_correlations: Dict[str, float] = {}
        self.current_positions: Dict[str, Dict[str, Any]] = {}

        logger.info("RiskManager initialized with advanced risk controls")

    def evaluate_trade_risk(
        self,
        ticker: str,
        proposed_position_change: int,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate risk for proposed trade and provide recommendations.

        Returns:
            Risk assessment with recommended position size and risk metrics
        """
        # Update regime detection
        if self.regime_detector:
            # Estimate volume (simplified)
            estimated_volume = abs(proposed_position_change) * 10
            self.regime_detector.update(current_price, estimated_volume)

        # Current portfolio state
        current_position = current_positions.get(ticker, {}).get('size', 0)
        new_position = current_position + proposed_position_change

        # Calculate optimal position size using Kelly criterion
        optimal_size = None
        if self.enable_kelly_sizing and self.kelly_calculator:
            optimal_size = self.kelly_calculator.get_optimal_position_size(
                portfolio_value, current_price, confidence
            )

        # Calculate risk metrics
        risk_metrics = self._calculate_position_risk(
            ticker, new_position, current_price, portfolio_value, current_positions
        )

        # Check position limits
        limit_violations = self._check_position_limits(
            ticker, new_position, current_price, portfolio_value, current_positions
        )

        # Apply regime-based adjustments
        regime_multiplier = 1.0
        if self.regime_detector:
            regime_multiplier = self.regime_detector.get_risk_multiplier()

        # Adjust proposed position based on risk factors
        adjusted_position_change = self._adjust_position_for_risk(
            proposed_position_change,
            optimal_size,
            current_position,
            risk_metrics,
            limit_violations,
            regime_multiplier
        )

        return {
            'original_position_change': proposed_position_change,
            'recommended_position_change': adjusted_position_change,
            'optimal_kelly_size': optimal_size,
            'risk_metrics': risk_metrics,
            'limit_violations': limit_violations,
            'regime_info': self.regime_detector.get_current_regime() if self.regime_detector else None,
            'confidence_adjusted': confidence * regime_multiplier
        }

    def _calculate_position_risk(
        self,
        ticker: str,
        position_size: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for position"""
        position_value = abs(position_size) * price
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0

        # Portfolio heat calculation
        total_exposure = sum(
            abs(pos.get('size', 0)) * pos.get('avg_price', price)
            for pos in current_positions.values()
        )
        total_exposure += position_value  # Add new position
        portfolio_heat = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Volatility estimate (simplified)
        volatility = 0.02  # Default 2% daily volatility

        # VaR calculation (simplified)
        var_95 = position_value * 1.645 * volatility  # 95% VaR

        return {
            'position_weight': position_weight,
            'portfolio_heat': portfolio_heat,
            'position_value': position_value,
            'volatility': volatility,
            'var_95': var_95,
            'leverage': portfolio_heat  # Simplified leverage calculation
        }

    def _check_position_limits(
        self,
        ticker: str,
        position_size: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any]
    ) -> List[str]:
        """Check for position limit violations"""
        violations = []

        # Maximum position size
        if abs(position_size) > self.position_limits.max_position_size:
            violations.append(f"Position size {abs(position_size)} exceeds maximum {self.position_limits.max_position_size}")

        # Minimum trade size
        if 0 < abs(position_size) < self.position_limits.min_trade_size:
            violations.append(f"Position size {abs(position_size)} below minimum {self.position_limits.min_trade_size}")

        # Portfolio heat limit
        position_value = abs(position_size) * price
        total_exposure = sum(
            abs(pos.get('size', 0)) * pos.get('avg_price', price)
            for pos in current_positions.values()
        )
        total_exposure += position_value
        portfolio_heat = total_exposure / portfolio_value if portfolio_value > 0 else 0

        if portfolio_heat > self.position_limits.max_portfolio_heat:
            violations.append(f"Portfolio heat {portfolio_heat:.2%} exceeds limit {self.position_limits.max_portfolio_heat:.2%}")

        return violations

    def _adjust_position_for_risk(
        self,
        proposed_change: int,
        optimal_kelly_size: Optional[int],
        current_position: int,
        risk_metrics: Dict[str, float],
        violations: List[str],
        regime_multiplier: float
    ) -> int:
        """Adjust position size based on risk factors"""
        adjusted_change = proposed_change

        # Apply regime adjustment
        adjusted_change = int(adjusted_change * regime_multiplier)

        # Apply Kelly criterion if available
        if optimal_kelly_size is not None:
            # If we have Kelly size, limit position to Kelly recommendation
            target_position = optimal_kelly_size
            max_change = target_position - current_position

            if abs(adjusted_change) > abs(max_change):
                adjusted_change = max_change

        # Apply hard limits to prevent violations
        if violations:
            # Reduce position size if there are violations
            reduction_factor = 0.5  # Reduce by 50% if there are violations
            adjusted_change = int(adjusted_change * reduction_factor)

        # Ensure we don't violate maximum position size
        new_position = current_position + adjusted_change
        if abs(new_position) > self.position_limits.max_position_size:
            max_allowed_change = self.position_limits.max_position_size - abs(current_position)
            adjusted_change = np.sign(adjusted_change) * max_allowed_change

        return adjusted_change

    def update_position(
        self,
        ticker: str,
        position_size: int,
        entry_price: float,
        entry_time: float
    ):
        """Update position tracking and set stop-loss"""
        self.current_positions[ticker] = {
            'size': position_size,
            'entry_price': entry_price,
            'entry_time': entry_time
        }

        # Set stop-loss for new position
        if position_size != 0:
            self.stop_loss_manager.set_position_stop(
                ticker, entry_price, position_size, entry_time
            )

    def check_stops(
        self,
        ticker: str,
        current_price: float,
        current_time: float
    ) -> Optional[str]:
        """Check if any stop-loss conditions are triggered"""
        return self.stop_loss_manager.update_stops(ticker, current_price, current_time)

    def close_position(self, ticker: str, exit_price: float, exit_time: float):
        """Handle position closure and update risk calculations"""
        if ticker in self.current_positions:
            position_info = self.current_positions[ticker]

            # Calculate PnL for Kelly criterion
            if self.kelly_calculator:
                entry_price = position_info['entry_price']
                position_size = position_info['size']
                pnl = (exit_price - entry_price) * position_size
                self.kelly_calculator.add_trade_result(pnl, position_size, entry_price)

            # Remove position tracking
            del self.current_positions[ticker]
            self.stop_loss_manager.remove_position_stop(ticker)

    def get_portfolio_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Get comprehensive portfolio risk metrics"""
        if not self.current_positions:
            return RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=0.0,
                max_individual_position=0.0,
                correlation_risk=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                cvar_95=0.0
            )

        # Calculate total exposure
        total_exposure = sum(
            abs(pos['size']) * pos['entry_price']
            for pos in self.current_positions.values()
        )

        # Find maximum individual position
        max_position = max(
            abs(pos['size']) * pos['entry_price']
            for pos in self.current_positions.values()
        ) if self.current_positions else 0.0

        # Simplified risk calculations
        portfolio_heat = total_exposure / portfolio_value if portfolio_value > 0 else 0
        volatility = 0.02  # Simplified
        var_95 = portfolio_value * 0.05  # Simplified 5% VaR

        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            max_individual_position=max_position,
            correlation_risk=0.0,  # Simplified
            volatility=volatility,
            sharpe_ratio=0.0,  # Would need return history
            max_drawdown=0.0,  # Would need portfolio history
            var_95=var_95,
            cvar_95=var_95 * 1.3  # Simplified CVaR
        )

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management summary"""
        return {
            'risk_manager_active': True,
            'kelly_criterion_enabled': self.enable_kelly_sizing,
            'regime_detection_enabled': self.enable_regime_detection,
            'current_regime': self.regime_detector.get_current_regime() if self.regime_detector else None,
            'active_positions': len(self.current_positions),
            'stop_losses_active': len(self.stop_loss_manager.position_stops),
            'position_limits': {
                'max_position_size': self.position_limits.max_position_size,
                'max_portfolio_heat': self.position_limits.max_portfolio_heat,
                'max_leverage': self.position_limits.max_leverage
            },
            'kelly_stats': {
                'trades_analyzed': len(self.kelly_calculator.trade_history) if self.kelly_calculator else 0,
                'current_kelly_fraction': self.kelly_calculator.calculate_kelly_fraction() if self.kelly_calculator else 0.0
            }
        }