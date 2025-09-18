"""
Baseline Trading Strategies

Implements various baseline trading strategies for benchmarking purposes.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from collections import deque

try:
    from .trading_environment import TradingEnvironment, OrderSide, OrderType, MarketState
except ImportError:
    from trading_environment import TradingEnvironment, OrderSide, OrderType, MarketState

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        """
        Generate trading signals based on current market state.
        
        Returns:
            List of tuples: (ticker, side, size, order_type, limit_price)
        """
        pass
    
    def on_trade_executed(self, trade_info: Dict[str, Any]):
        """Callback when a trade is executed"""
        pass
    
    def reset(self):
        """Reset strategy state"""
        self.portfolio_history = []


class RuleBasedTrader(BaseStrategy):
    """
    Simple rule-based strategy:
    - Buy after touchdowns (price momentum up)
    - Sell after field goals or negative events
    - Hold positions for limited time
    """
    
    def __init__(
        self,
        touchdown_buy_size: int = 50,
        field_goal_sell_size: int = 30,
        hold_duration_minutes: int = 3,
        min_price_move: float = 0.02,
        max_position_size: int = 200
    ):
        super().__init__("RuleBasedTrader")
        self.touchdown_buy_size = touchdown_buy_size
        self.field_goal_sell_size = field_goal_sell_size
        self.hold_duration_minutes = hold_duration_minutes
        self.min_price_move = min_price_move
        self.max_position_size = max_position_size
        
        # Track recent events and positions
        self.recent_events: deque = deque(maxlen=10)
        self.position_timestamps: Dict[str, float] = {}
        self.last_prices: Dict[str, float] = {}
        
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        
        signals = []
        current_positions = env.get_position_summary()
        
        for ticker, market_state in market_data.items():
            if not market_state:
                continue
                
            # Check for price movements indicating scoring events
            price_change = 0.0
            if ticker in self.last_prices:
                price_change = market_state.mid_price - self.last_prices[ticker]
            
            self.last_prices[ticker] = market_state.mid_price
            
            # Rule 1: Buy on significant upward price movement (touchdown indicator)
            if price_change > self.min_price_move:
                current_size = current_positions.get(ticker, {}).get('size', 0)
                if current_size < self.max_position_size:
                    buy_size = min(self.touchdown_buy_size, self.max_position_size - current_size)
                    if buy_size > 0:
                        signals.append((ticker, OrderSide.BUY, buy_size, OrderType.MARKET, None))
                        self.position_timestamps[ticker] = env.current_timestamp
                        logger.debug(f"Rule signal: BUY {buy_size} {ticker} on price jump {price_change:.4f}")
            
            # Rule 2: Sell on negative price movement or hold time exceeded
            elif price_change < -self.min_price_move or self._should_exit_position(ticker, env.current_timestamp):
                current_size = current_positions.get(ticker, {}).get('size', 0)
                if current_size > 0:
                    sell_size = min(self.field_goal_sell_size, current_size)
                    signals.append((ticker, OrderSide.SELL, sell_size, OrderType.MARKET, None))
                    logger.debug(f"Rule signal: SELL {sell_size} {ticker} on price drop {price_change:.4f} or time limit")
        
        return signals
    
    def _should_exit_position(self, ticker: str, current_timestamp: float) -> bool:
        """Check if position should be exited based on hold duration"""
        if ticker not in self.position_timestamps:
            return False
            
        hold_time = current_timestamp - self.position_timestamps[ticker]
        return hold_time > (self.hold_duration_minutes * 60)  # Convert minutes to seconds


class StatisticalTrader(BaseStrategy):
    """
    Statistical trading strategy using:
    - Mean reversion signals
    - Momentum indicators  
    - Volatility-adjusted position sizing
    - RSI-like indicators
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        rsi_period: int = 14,
        bollinger_std: float = 2.0,
        mean_reversion_threshold: float = 0.03,
        momentum_threshold: float = 0.02,
        base_position_size: int = 50,
        max_position_size: int = 300
    ):
        super().__init__("StatisticalTrader")
        self.lookback_window = lookback_window
        self.rsi_period = rsi_period
        self.bollinger_std = bollinger_std
        self.mean_reversion_threshold = mean_reversion_threshold
        self.momentum_threshold = momentum_threshold
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        
        # Price history for calculations
        self.price_history: Dict[str, deque] = {}
        
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        
        signals = []
        current_positions = env.get_position_summary()
        
        for ticker, market_state in market_data.items():
            if not market_state:
                continue
                
            # Update price history
            if ticker not in self.price_history:
                self.price_history[ticker] = deque(maxlen=self.lookback_window)
            
            self.price_history[ticker].append(market_state.mid_price)
            
            # Need sufficient price history
            if len(self.price_history[ticker]) < self.rsi_period:
                continue
                
            prices = np.array(self.price_history[ticker])
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(prices)
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(prices)
            momentum = self._calculate_momentum(prices)
            volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0.01
            
            current_price = market_state.mid_price
            current_size = current_positions.get(ticker, {}).get('size', 0)
            
            # Position sizing based on volatility
            vol_adjusted_size = max(10, int(self.base_position_size / (1 + volatility * 10)))
            
            # Mean reversion signals
            if current_price > bollinger_upper and rsi > 70:
                # Overbought - sell signal
                if current_size > 0:
                    sell_size = min(vol_adjusted_size, current_size)
                    signals.append((ticker, OrderSide.SELL, sell_size, OrderType.MARKET, None))
                    logger.debug(f"Statistical signal: SELL {sell_size} {ticker} (overbought, RSI={rsi:.2f})")
                    
            elif current_price < bollinger_lower and rsi < 30:
                # Oversold - buy signal
                if current_size < self.max_position_size:
                    buy_size = min(vol_adjusted_size, self.max_position_size - current_size)
                    signals.append((ticker, OrderSide.BUY, buy_size, OrderType.MARKET, None))
                    logger.debug(f"Statistical signal: BUY {buy_size} {ticker} (oversold, RSI={rsi:.2f})")
            
            # Momentum signals
            elif momentum > self.momentum_threshold and rsi < 60:
                # Strong momentum - trend following
                if current_size < self.max_position_size:
                    buy_size = min(vol_adjusted_size, self.max_position_size - current_size)
                    signals.append((ticker, OrderSide.BUY, buy_size, OrderType.MARKET, None))
                    logger.debug(f"Statistical signal: BUY {buy_size} {ticker} (momentum={momentum:.4f})")
                    
            elif momentum < -self.momentum_threshold and rsi > 40:
                # Negative momentum
                if current_size > 0:
                    sell_size = min(vol_adjusted_size, current_size)
                    signals.append((ticker, OrderSide.SELL, sell_size, OrderType.MARKET, None))
                    logger.debug(f"Statistical signal: SELL {sell_size} {ticker} (neg momentum={momentum:.4f})")
        
        return signals
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < 20:
            mean_price = np.mean(prices)
            std_price = np.std(prices) if len(prices) > 1 else 0.01
        else:
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:])
            
        upper_band = mean_price + self.bollinger_std * std_price
        lower_band = mean_price - self.bollinger_std * std_price
        
        return upper_band, lower_band
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        if len(prices) < 10:
            return 0.0
            
        recent_avg = np.mean(prices[-5:])
        older_avg = np.mean(prices[-10:-5])
        
        if older_avg == 0:
            return 0.0
            
        momentum = (recent_avg - older_avg) / older_avg
        return momentum


class RandomTrader(BaseStrategy):
    """
    Random trading strategy for baseline comparison.
    Generates random buy/sell signals with controlled risk.
    """
    
    def __init__(
        self,
        trade_probability: float = 0.05,
        max_position_size: int = 100,
        position_size_range: Tuple[int, int] = (10, 50),
        random_seed: Optional[int] = None
    ):
        super().__init__("RandomTrader")
        self.trade_probability = trade_probability
        self.max_position_size = max_position_size
        self.position_size_range = position_size_range
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        
        signals = []
        current_positions = env.get_position_summary()
        
        for ticker, market_state in market_data.items():
            if not market_state:
                continue
                
            # Random decision to trade
            if np.random.random() < self.trade_probability:
                current_size = current_positions.get(ticker, {}).get('size', 0)
                
                # Random position size
                trade_size = np.random.randint(self.position_size_range[0], self.position_size_range[1] + 1)
                
                # Random side with some logic
                if current_size == 0:
                    # No position - can only buy
                    side = OrderSide.BUY
                    final_size = min(trade_size, self.max_position_size)
                elif current_size >= self.max_position_size:
                    # Max position - can only sell
                    side = OrderSide.SELL
                    final_size = min(trade_size, current_size)
                else:
                    # Can buy or sell
                    side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL
                    if side == OrderSide.BUY:
                        final_size = min(trade_size, self.max_position_size - current_size)
                    else:
                        final_size = min(trade_size, current_size)
                
                if final_size > 0:
                    signals.append((ticker, side, final_size, OrderType.MARKET, None))
                    logger.debug(f"Random signal: {side.value} {final_size} {ticker}")
        
        return signals


class BuyAndHoldTrader(BaseStrategy):
    """
    Buy and hold strategy - buys at the beginning and holds until the end.
    Useful as a passive benchmark.
    """
    
    def __init__(
        self, 
        initial_position_size: int = 100,
        buy_spread_minutes: int = 5
    ):
        super().__init__("BuyAndHoldTrader")
        self.initial_position_size = initial_position_size
        self.buy_spread_minutes = buy_spread_minutes
        self.initial_purchases_made: Dict[str, bool] = {}
        self.start_timestamp: Optional[float] = None
        
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        
        signals = []
        
        # Set start timestamp on first call
        if self.start_timestamp is None:
            self.start_timestamp = env.current_timestamp
        
        # Only buy in the first few minutes
        time_since_start = env.current_timestamp - self.start_timestamp
        if time_since_start > (self.buy_spread_minutes * 60):
            return signals
        
        current_positions = env.get_position_summary()
        
        for ticker, market_state in market_data.items():
            if not market_state:
                continue
                
            # Make initial purchase if not already done
            if ticker not in self.initial_purchases_made:
                current_size = current_positions.get(ticker, {}).get('size', 0)
                if current_size == 0:
                    signals.append((ticker, OrderSide.BUY, self.initial_position_size, OrderType.MARKET, None))
                    self.initial_purchases_made[ticker] = True
                    logger.debug(f"Buy and hold: BUY {self.initial_position_size} {ticker}")
        
        return signals
    
    def reset(self):
        super().reset()
        self.initial_purchases_made = {}
        self.start_timestamp = None


class MomentumFollower(BaseStrategy):
    """
    Momentum following strategy that trades based on sustained price movements.
    """
    
    def __init__(
        self,
        momentum_lookback: int = 5,
        momentum_threshold: float = 0.015,
        position_size: int = 40,
        max_position_size: int = 200,
        stop_loss_pct: float = 0.05
    ):
        super().__init__("MomentumFollower")
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        
        self.price_history: Dict[str, deque] = {}
        self.entry_prices: Dict[str, float] = {}
        
    def generate_signals(
        self, 
        env: TradingEnvironment, 
        market_data: Dict[str, MarketState],
        game_events: Optional[List[Dict]] = None
    ) -> List[Tuple[str, OrderSide, int, OrderType, Optional[float]]]:
        
        signals = []
        current_positions = env.get_position_summary()
        
        for ticker, market_state in market_data.items():
            if not market_state:
                continue
                
            # Update price history
            if ticker not in self.price_history:
                self.price_history[ticker] = deque(maxlen=self.momentum_lookback * 2)
            
            self.price_history[ticker].append(market_state.mid_price)
            
            if len(self.price_history[ticker]) < self.momentum_lookback:
                continue
                
            prices = np.array(self.price_history[ticker])
            current_price = market_state.mid_price
            current_size = current_positions.get(ticker, {}).get('size', 0)
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices)
            
            # Stop loss check
            if current_size > 0 and ticker in self.entry_prices:
                price_change = (current_price - self.entry_prices[ticker]) / self.entry_prices[ticker]
                if price_change < -self.stop_loss_pct:
                    signals.append((ticker, OrderSide.SELL, current_size, OrderType.MARKET, None))
                    logger.debug(f"Momentum: STOP LOSS {current_size} {ticker} at {price_change:.2%}")
                    del self.entry_prices[ticker]
                    continue
            
            # Momentum signals
            if momentum > self.momentum_threshold and current_size < self.max_position_size:
                # Strong upward momentum
                buy_size = min(self.position_size, self.max_position_size - current_size)
                signals.append((ticker, OrderSide.BUY, buy_size, OrderType.MARKET, None))
                self.entry_prices[ticker] = current_price
                logger.debug(f"Momentum: BUY {buy_size} {ticker} (momentum={momentum:.4f})")
                
            elif momentum < -self.momentum_threshold and current_size > 0:
                # Strong downward momentum
                sell_size = min(self.position_size, current_size)
                signals.append((ticker, OrderSide.SELL, sell_size, OrderType.MARKET, None))
                logger.debug(f"Momentum: SELL {sell_size} {ticker} (momentum={momentum:.4f})")
        
        return signals
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum over lookback period"""
        if len(prices) < self.momentum_lookback:
            return 0.0
            
        start_price = prices[-self.momentum_lookback]
        end_price = prices[-1]
        
        if start_price == 0:
            return 0.0
            
        momentum = (end_price - start_price) / start_price
        return momentum
    
    def reset(self):
        super().reset()
        self.price_history = {}
        self.entry_prices = {}