"""
ML-Based Trading Strategy

Integrates play description analysis models with the backtesting framework.
Uses transformer models to predict play outcomes and price impacts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from datetime import datetime

try:
    from .trading_environment import TradingEnvironment, OrderSide, OrderType, MarketState
    from .strategies import BaseStrategy
except ImportError:
    from trading_environment import TradingEnvironment, OrderSide, OrderType, MarketState
    from strategies import BaseStrategy

# Try to import ML components
try:
    import sys
    import os
    # Add ML directory to path
    ml_path = os.path.join(os.path.dirname(__file__), '..', 'ml')
    if ml_path not in sys.path:
        sys.path.append(ml_path)

    # Try both relative and absolute imports
    try:
        from inference.play_predictor import PlayPredictor
        from text_processor import PlayTextProcessor
    except ImportError:
        # Fallback for different import paths
        try:
            from ..ml.inference.play_predictor import PlayPredictor
            from ..ml.text_processor import PlayTextProcessor
        except ImportError:
            # Final fallback - mark as unavailable but don't crash
            raise ImportError("ML components not found in any expected location")

    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLTradingStrategy(BaseStrategy):
    """
    ML-powered trading strategy that uses play description analysis
    to make trading decisions based on predicted outcomes and price impacts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        min_price_impact_threshold: float = 0.02,
        max_position_size: int = 200,
        position_scaling_factor: float = 1.0,
        use_synthetic_data: bool = True,
        fallback_to_momentum: bool = True
    ):
        super().__init__("MLTradingStrategy")

        self.confidence_threshold = confidence_threshold
        self.min_price_impact_threshold = min_price_impact_threshold
        self.max_position_size = max_position_size
        self.position_scaling_factor = position_scaling_factor
        self.use_synthetic_data = use_synthetic_data
        self.fallback_to_momentum = fallback_to_momentum

        # Initialize ML components if available
        self.ml_available = ML_AVAILABLE
        self.predictor = None
        self.text_processor = None

        if self.ml_available:
            try:
                self.text_processor = PlayTextProcessor()
                if model_path and os.path.exists(model_path):
                    self.predictor = PlayPredictor(model_path=model_path)
                    logger.info("ML components loaded successfully")
                else:
                    logger.warning("Model path not provided or doesn't exist, using synthetic predictions")
            except Exception as e:
                logger.warning(f"Failed to load ML components: {e}")
                self.ml_available = False

        # Fallback momentum tracking
        self.price_history: Dict[str, deque] = {}
        self.momentum_lookback = 5
        self.momentum_threshold = 0.015

        # Game state tracking
        self.game_events_history: List[Dict] = []
        self.last_prediction_time: Dict[str, float] = {}
        self.prediction_cache: Dict[str, Dict] = {}

    def generate_signals(
        self,
        env: TradingEnvironment,
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:

        signals = []
        current_positions = env.get_position_summary()

        # Update game events history
        if game_events:
            self.game_events_history.extend(game_events)
            # Keep only recent events (last 50)
            self.game_events_history = self.game_events_history[-50:]

        for ticker, market_state in market_data.items():
            if not market_state:
                continue

            # Update price history for fallback momentum strategy
            if ticker not in self.price_history:
                self.price_history[ticker] = deque(maxlen=self.momentum_lookback * 2)
            self.price_history[ticker].append(market_state.mid_price)

            current_size = current_positions.get(ticker, {}).get('size', 0)

            # Try ML prediction first
            if self.ml_available:
                ml_signals = self._generate_ml_signals(ticker, market_state, env.current_timestamp)
                if ml_signals:
                    for signal in ml_signals:
                        side, size = signal
                        if side == OrderSide.BUY and current_size < self.max_position_size:
                            final_size = min(size, self.max_position_size - current_size)
                            if final_size > 0:
                                signals.append((ticker, side, final_size, OrderType.MARKET, None))
                                logger.debug(f"ML signal: BUY {final_size} {ticker}")
                        elif side == OrderSide.SELL and current_size > 0:
                            final_size = min(size, current_size)
                            if final_size > 0:
                                signals.append((ticker, side, final_size, OrderType.MARKET, None))
                                logger.debug(f"ML signal: SELL {final_size} {ticker}")
                    continue

            # Fallback to momentum strategy if ML not available or no confident predictions
            if self.fallback_to_momentum and len(self.price_history[ticker]) >= self.momentum_lookback:
                momentum_signals = self._generate_momentum_signals(ticker, market_state, current_size)
                signals.extend(momentum_signals)

        return signals

    def _generate_ml_signals(
        self,
        ticker: str,
        market_state: MarketState,
        current_timestamp: float
    ) -> List[Tuple[OrderSide, int]]:
        """Generate trading signals using ML predictions"""

        signals = []

        # Check if we should make a new prediction (rate limiting)
        if ticker in self.last_prediction_time:
            time_since_last = current_timestamp - self.last_prediction_time[ticker]
            if time_since_last < 30:  # Don't predict more than once every 30 seconds
                return signals

        try:
            # Generate synthetic play description for demonstration
            play_description = self._create_synthetic_play_description(ticker, market_state)

            # Create numerical features from market state
            numerical_features = np.array([
                market_state.mid_price,
                market_state.bid_price,
                market_state.ask_price,
                market_state.volume / 1000.0,  # Normalize volume
                market_state.open_interest / 10000.0,  # Normalize open interest
                len(self.game_events_history),  # Number of recent events
            ])

            if self.predictor and not self.use_synthetic_data:
                # Use actual ML model
                prediction = self.predictor.predict_play(
                    play_description=play_description,
                    numerical_features=numerical_features,
                    return_feature_importance=False
                )

                confidence = prediction.get('confidence', 0.5)
                predicted_price_impact = prediction.get('price_impact_prediction', 0.0)
                predicted_outcome = prediction.get('outcome_prediction', 0.5)

            else:
                # Use synthetic predictions for demonstration
                prediction = self._generate_synthetic_prediction(ticker, market_state, numerical_features)
                confidence = prediction['confidence']
                predicted_price_impact = prediction['price_impact_prediction']
                predicted_outcome = prediction['outcome_prediction']

            # Cache the prediction
            self.prediction_cache[ticker] = {
                'confidence': confidence,
                'price_impact': predicted_price_impact,
                'outcome': predicted_outcome,
                'timestamp': current_timestamp
            }
            self.last_prediction_time[ticker] = current_timestamp

            # Generate signals based on predictions
            if confidence > self.confidence_threshold:
                position_size = int(50 * confidence * self.position_scaling_factor)

                # Positive price impact prediction -> buy signal
                if predicted_price_impact > self.min_price_impact_threshold:
                    signals.append((OrderSide.BUY, position_size))
                    logger.debug(f"ML: Confident BUY signal for {ticker} (confidence={confidence:.3f}, impact={predicted_price_impact:.4f})")

                # Negative price impact prediction -> sell signal
                elif predicted_price_impact < -self.min_price_impact_threshold:
                    signals.append((OrderSide.SELL, position_size))
                    logger.debug(f"ML: Confident SELL signal for {ticker} (confidence={confidence:.3f}, impact={predicted_price_impact:.4f})")

        except Exception as e:
            logger.warning(f"ML prediction failed for {ticker}: {e}")

        return signals

    def _generate_momentum_signals(
        self,
        ticker: str,
        market_state: MarketState,
        current_size: int
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        """Fallback momentum-based signals"""

        signals = []
        prices = np.array(self.price_history[ticker])

        if len(prices) < self.momentum_lookback:
            return signals

        # Calculate simple momentum
        momentum = (prices[-1] - prices[-self.momentum_lookback]) / prices[-self.momentum_lookback]

        base_size = 25

        if momentum > self.momentum_threshold and current_size < self.max_position_size:
            buy_size = min(base_size, self.max_position_size - current_size)
            signals.append((ticker, OrderSide.BUY, buy_size, OrderType.MARKET, None))
            logger.debug(f"Momentum fallback: BUY {buy_size} {ticker} (momentum={momentum:.4f})")

        elif momentum < -self.momentum_threshold and current_size > 0:
            sell_size = min(base_size, current_size)
            signals.append((ticker, OrderSide.SELL, sell_size, OrderType.MARKET, None))
            logger.debug(f"Momentum fallback: SELL {sell_size} {ticker} (momentum={momentum:.4f})")

        return signals

    def _create_synthetic_play_description(self, ticker: str, market_state: MarketState) -> str:
        """Create synthetic play descriptions for demonstration"""

        # Extract team information from ticker if possible
        teams = ["Cowboys", "Eagles", "Patriots", "Chiefs", "49ers", "Steelers"]
        team = np.random.choice(teams)

        # Create realistic play descriptions based on market movement
        recent_prices = list(self.price_history.get(ticker, [market_state.mid_price]))
        price_trend = "rising" if len(recent_prices) > 1 and recent_prices[-1] > recent_prices[0] else "falling"

        if price_trend == "rising":
            plays = [
                f"{team} touchdown run from the 5-yard line",
                f"{team} completes 35-yard pass for first down",
                f"{team} intercepts pass in red zone",
                f"{team} recovers fumble at opponent 20-yard line",
                f"{team} kicks successful field goal from 35 yards"
            ]
        else:
            plays = [
                f"{team} fumbles at the 25-yard line",
                f"{team} throws interception in red zone",
                f"{team} misses field goal from 40 yards",
                f"{team} penalized for holding on 3rd down",
                f"{team} punts after three-and-out"
            ]

        return np.random.choice(plays)

    def _generate_synthetic_prediction(
        self,
        ticker: str,
        market_state: MarketState,
        numerical_features: np.ndarray
    ) -> Dict[str, float]:
        """Generate synthetic ML predictions for demonstration"""

        # Use market momentum and volatility to create realistic predictions
        recent_prices = list(self.price_history.get(ticker, [market_state.mid_price]))

        if len(recent_prices) > 1:
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            volatility = np.std(recent_prices) if len(recent_prices) > 2 else 0.01
        else:
            price_change = 0.0
            volatility = 0.01

        # Simulate model confidence based on market conditions
        base_confidence = 0.6 + min(0.3, abs(price_change) * 10)  # Higher confidence with more movement
        confidence = max(0.3, base_confidence - volatility * 5)  # Lower confidence with high volatility

        # Simulate price impact prediction with some noise
        trend_factor = np.tanh(price_change * 20)  # Sigmoid-like trend following
        noise = np.random.normal(0, 0.01)
        predicted_price_impact = trend_factor * 0.03 + noise

        # Simulate binary outcome prediction
        outcome_prob = 0.5 + trend_factor * 0.2 + np.random.normal(0, 0.1)
        outcome_prob = np.clip(outcome_prob, 0.1, 0.9)

        return {
            'confidence': confidence,
            'price_impact_prediction': predicted_price_impact,
            'outcome_prediction': outcome_prob
        }

    def get_strategy_state(self) -> Dict[str, Any]:
        """Return current strategy state for analysis"""
        return {
            'ml_available': self.ml_available,
            'prediction_cache': self.prediction_cache.copy(),
            'recent_events_count': len(self.game_events_history),
            'tickers_tracked': list(self.price_history.keys())
        }

    def reset(self):
        """Reset strategy state"""
        super().reset()
        self.price_history = {}
        self.game_events_history = []
        self.last_prediction_time = {}
        self.prediction_cache = {}


class EnsembleMLStrategy(BaseStrategy):
    """
    Ensemble strategy that combines multiple ML models and baseline strategies
    for more robust predictions.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        weights: Optional[List[float]] = None,
        min_agreement_threshold: float = 0.6
    ):
        super().__init__("EnsembleMLStrategy")

        if strategies is None:
            # Default ensemble with ML and momentum
            strategies = [
                MLTradingStrategy(confidence_threshold=0.6),
                # Could add other strategies here
            ]

        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.min_agreement_threshold = min_agreement_threshold

        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")

    def generate_signals(
        self,
        env: TradingEnvironment,
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:

        # Collect signals from all strategies
        all_signals = {}

        for i, strategy in enumerate(self.strategies):
            try:
                strategy_signals = strategy.generate_signals(env, market_data, game_events)
                weight = self.weights[i]

                for ticker, side, size, order_type, limit_price in strategy_signals:
                    if ticker not in all_signals:
                        all_signals[ticker] = {'buy_votes': 0, 'sell_votes': 0, 'buy_size': 0, 'sell_size': 0}

                    if side == OrderSide.BUY:
                        all_signals[ticker]['buy_votes'] += weight
                        all_signals[ticker]['buy_size'] += size * weight
                    else:
                        all_signals[ticker]['sell_votes'] += weight
                        all_signals[ticker]['sell_size'] += size * weight

            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")

        # Generate ensemble signals
        ensemble_signals = []

        for ticker, votes in all_signals.items():
            total_buy_weight = votes['buy_votes']
            total_sell_weight = votes['sell_votes']

            # Check for sufficient agreement
            if total_buy_weight > self.min_agreement_threshold and total_buy_weight > total_sell_weight:
                avg_size = int(votes['buy_size'] / total_buy_weight) if total_buy_weight > 0 else 0
                if avg_size > 0:
                    ensemble_signals.append((ticker, OrderSide.BUY, avg_size, OrderType.MARKET, None))
                    logger.debug(f"Ensemble: BUY {avg_size} {ticker} (agreement={total_buy_weight:.2f})")

            elif total_sell_weight > self.min_agreement_threshold and total_sell_weight > total_buy_weight:
                avg_size = int(votes['sell_size'] / total_sell_weight) if total_sell_weight > 0 else 0
                if avg_size > 0:
                    ensemble_signals.append((ticker, OrderSide.SELL, avg_size, OrderType.MARKET, None))
                    logger.debug(f"Ensemble: SELL {avg_size} {ticker} (agreement={total_sell_weight:.2f})")

        return ensemble_signals

    def reset(self):
        super().reset()
        for strategy in self.strategies:
            strategy.reset()