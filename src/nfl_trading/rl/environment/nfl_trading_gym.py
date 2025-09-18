"""
NFL Trading Gym Environment

OpenAI Gym-compatible environment for training RL agents on NFL betting markets.
Integrates with the existing backtesting framework.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

# Try to import backtesting components
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backtesting'))
    from trading_environment import TradingEnvironment, MarketState, OrderSide, OrderType
except ImportError:
    logging.warning("Backtesting components not available")

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Current state of an NFL game"""
    quarter: int
    time_remaining: int  # seconds
    score_home: int
    score_away: int
    down: int
    distance: int
    field_position: int
    possession: str  # 'home' or 'away'
    recent_events: List[str]


@dataclass
class MarketFeatures:
    """Current market features"""
    mid_price: float
    bid_ask_spread: float
    volume: float
    price_momentum_5m: float
    price_momentum_1m: float
    volatility: float
    order_book_imbalance: float


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    position_size: int
    unrealized_pnl: float
    position_duration: int  # minutes
    avg_entry_price: float
    risk_exposure: float


class NFLTradingGym(gym.Env):
    """
    NFL Trading Gym Environment

    State Space: Game state + Market features + Portfolio state
    Action Space: Position adjustment (-1 to +1, or discrete actions)
    Reward: Risk-adjusted PnL with transaction costs
    """

    def __init__(
        self,
        data_files: List[str],
        initial_capital: float = 10000.0,
        max_position_size: int = 500,
        transaction_cost_bps: float = 7.0,
        risk_penalty_factor: float = 0.1,
        reward_lookback_minutes: int = 5,
        action_type: str = "continuous",  # "continuous" or "discrete"
        state_history_length: int = 10,
        normalize_features: bool = True,
        include_game_features: bool = True,
        warmup_minutes: int = 2
    ):
        super().__init__()

        self.data_files = data_files
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost_bps = transaction_cost_bps / 10000.0  # Convert to decimal
        self.risk_penalty_factor = risk_penalty_factor
        self.reward_lookback_minutes = reward_lookback_minutes
        self.action_type = action_type
        self.state_history_length = state_history_length
        self.normalize_features = normalize_features
        self.include_game_features = include_game_features
        self.warmup_minutes = warmup_minutes

        # Initialize trading environment
        self.trading_env = TradingEnvironment(initial_capital=initial_capital)

        # Define action space
        if action_type == "continuous":
            # Continuous action: position adjustment as fraction of max position
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            # Discrete actions: [hold, small_buy, large_buy, small_sell, large_sell]
            self.action_space = spaces.Discrete(5)

        # Define observation space dimensions
        market_features_dim = 7  # mid_price, spread, volume, momentum_5m, momentum_1m, volatility, imbalance
        portfolio_features_dim = 6  # cash, position, pnl, duration, entry_price, risk
        game_features_dim = 9 if include_game_features else 0  # quarter, time, scores, down, distance, etc.

        # Total features per timestep
        features_per_timestep = market_features_dim + portfolio_features_dim + game_features_dim

        # State includes history
        total_features = features_per_timestep * state_history_length

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # Internal state
        self.current_game_idx = 0
        self.current_ticker = None
        self.market_data_generator = None
        self.episode_rewards = []
        self.episode_actions = []
        self.state_history = deque(maxlen=state_history_length)
        self.price_history = deque(maxlen=100)
        self.last_position_size = 0
        self.episode_start_time = None
        self.episode_end_time = None

        # Normalization statistics (will be computed during first episode)
        self.feature_means = None
        self.feature_stds = None

        # Game state tracking
        self.game_state = GameState(0, 3600, 0, 0, 1, 10, 50, 'home', [])

        # Performance tracking
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'transaction_costs': 0.0,
            'num_trades': 0,
            'win_rate': 0.0
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Select random game for this episode
        if self.data_files:
            self.current_game_idx = np.random.randint(len(self.data_files))
            game_file = self.data_files[self.current_game_idx]
        else:
            raise ValueError("No data files provided")

        # Reset trading environment
        self.trading_env = TradingEnvironment(initial_capital=self.initial_capital)

        try:
            # Load market data
            self.trading_env.load_market_data(game_file)

            # Get first ticker
            if self.trading_env.market_data:
                self.current_ticker = list(self.trading_env.market_data.keys())[0]
            else:
                raise ValueError(f"No market data loaded from {game_file}")

        except Exception as e:
            logger.error(f"Failed to load game data: {e}")
            # Use mock data for testing
            self._create_mock_episode()

        # Reset internal state
        self.episode_rewards = []
        self.episode_actions = []
        self.state_history.clear()
        self.price_history.clear()
        self.last_position_size = 0

        # Initialize game state
        self.game_state = GameState(1, 3600, 0, 0, 1, 10, 50, 'home', [])

        # Get initial observation
        initial_obs = self._get_observation()

        # Initialize state history with current observation
        feature_vector = self._extract_features()
        for _ in range(self.state_history_length):
            self.state_history.append(feature_vector)

        # Reset episode tracking
        self.episode_start_time = self.trading_env.current_timestamp
        self.episode_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'transaction_costs': 0.0,
            'num_trades': 0,
            'win_rate': 0.0
        }

        info = {
            'game_file': game_file if 'game_file' in locals() else 'mock_data',
            'ticker': self.current_ticker,
            'episode_length': 0
        }

        return initial_obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state, reward, done, info"""

        # Convert action to position change
        position_change = self._action_to_position_change(action)

        # Execute trading action
        reward, transaction_cost, num_trades = self._execute_action(position_change)

        # Advance time in trading environment
        self._advance_time()

        # Update game state (simplified simulation)
        self._update_game_state()

        # Get new observation
        obs = self._get_observation()

        # Check if episode is done
        done = self._is_episode_done()
        truncated = False

        # Track metrics
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_metrics['transaction_costs'] += transaction_cost
        self.episode_metrics['num_trades'] += num_trades

        # Calculate episode metrics if done
        if done:
            self._calculate_episode_metrics()

        info = {
            'position_size': self._get_current_position_size(),
            'portfolio_value': self.trading_env.get_portfolio_value(),
            'transaction_cost': transaction_cost,
            'game_state': self.game_state,
            'episode_metrics': self.episode_metrics if done else {}
        }

        return obs, reward, done, truncated, info

    def _action_to_position_change(self, action: Union[int, np.ndarray]) -> int:
        """Convert RL action to position size change"""

        current_position = self._get_current_position_size()

        if self.action_type == "continuous":
            # Continuous action: -1 to +1 scaled to max position change
            if isinstance(action, np.ndarray):
                action_value = action[0]
            else:
                action_value = action

            # Scale action to position change
            max_change = min(100, self.max_position_size - abs(current_position))
            position_change = int(action_value * max_change)

        else:
            # Discrete actions
            action_map = {
                0: 0,      # Hold
                1: 25,     # Small buy
                2: 50,     # Large buy
                3: -25,    # Small sell
                4: -50     # Large sell
            }
            position_change = action_map.get(action, 0)

        # Ensure position change respects limits
        new_position = current_position + position_change
        if abs(new_position) > self.max_position_size:
            position_change = np.sign(position_change) * (self.max_position_size - abs(current_position))

        return position_change

    def _execute_action(self, position_change: int) -> Tuple[float, float, int]:
        """Execute trading action and return reward, transaction cost, number of trades"""

        if abs(position_change) < 5:  # Minimum trade size
            return 0.0, 0.0, 0

        current_market_state = self.trading_env.get_current_market_state(self.current_ticker)
        if not current_market_state:
            return 0.0, 0.0, 0

        # Calculate transaction cost
        transaction_cost = abs(position_change) * current_market_state.mid_price * self.transaction_cost_bps

        # Execute trade
        num_trades = 0
        try:
            if position_change > 0:
                self.trading_env.place_order(
                    ticker=self.current_ticker,
                    side=OrderSide.BUY,
                    size=abs(position_change),
                    order_type=OrderType.MARKET
                )
                num_trades = 1
            elif position_change < 0:
                self.trading_env.place_order(
                    ticker=self.current_ticker,
                    side=OrderSide.SELL,
                    size=abs(position_change),
                    order_type=OrderType.MARKET
                )
                num_trades = 1

        except Exception as e:
            logger.warning(f"Trade execution failed: {e}")
            transaction_cost = 0.0
            num_trades = 0

        # Calculate reward
        reward = self._calculate_reward(transaction_cost)

        return reward, transaction_cost, num_trades

    def _calculate_reward(self, transaction_cost: float) -> float:
        """Calculate reward based on PnL, risk, and transaction costs"""

        # Get current portfolio metrics
        current_portfolio_value = self.trading_env.get_portfolio_value()

        # Calculate PnL change
        if not hasattr(self, '_last_portfolio_value'):
            self._last_portfolio_value = self.initial_capital

        pnl_change = current_portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = current_portfolio_value

        # Base reward from PnL
        reward = pnl_change

        # Subtract transaction costs
        reward -= transaction_cost

        # Risk penalty based on position size and volatility
        current_position = self._get_current_position_size()
        if len(self.price_history) > 10:
            price_volatility = np.std(list(self.price_history)[-10:])
            risk_penalty = self.risk_penalty_factor * abs(current_position) * price_volatility
            reward -= risk_penalty

        # Bonus for profitable position sizing
        if len(self.price_history) > 5:
            price_momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            if (current_position > 0 and price_momentum > 0) or (current_position < 0 and price_momentum < 0):
                reward += abs(current_position) * 0.001  # Small momentum bonus

        # Scale reward for better learning
        reward = reward / 100.0  # Scale down for stability

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""

        # Extract current features
        current_features = self._extract_features()

        # Add to history
        self.state_history.append(current_features)

        # Flatten history into observation vector
        obs = np.concatenate(list(self.state_history))

        # Normalize if enabled
        if self.normalize_features:
            obs = self._normalize_features(obs)

        return obs.astype(np.float32)

    def _extract_features(self) -> np.ndarray:
        """Extract feature vector from current state"""

        features = []

        # Market features
        market_state = self.trading_env.get_current_market_state(self.current_ticker)
        if market_state:
            # Price features
            features.extend([
                market_state.mid_price,
                market_state.ask_price - market_state.bid_price,  # spread
                market_state.volume / 1000.0,  # normalized volume
            ])

            # Update price history
            self.price_history.append(market_state.mid_price)

            # Momentum features
            if len(self.price_history) >= 5:
                momentum_5m = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
                features.append(momentum_5m)
            else:
                features.append(0.0)

            if len(self.price_history) >= 2:
                momentum_1m = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
                features.append(momentum_1m)
            else:
                features.append(0.0)

            # Volatility
            if len(self.price_history) >= 10:
                volatility = np.std(list(self.price_history)[-10:])
                features.append(volatility)
            else:
                features.append(0.01)

            # Order book imbalance (simplified)
            bid_size = market_state.volume * 0.4  # Assume 40% on bid
            ask_size = market_state.volume * 0.6  # Assume 60% on ask
            imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
            features.append(imbalance)

        else:
            # Default values if no market data
            features.extend([0.5, 0.02, 100.0, 0.0, 0.0, 0.01, 0.0])

        # Portfolio features
        current_position = self._get_current_position_size()
        portfolio_value = self.trading_env.get_portfolio_value()
        cash = self.trading_env.cash

        features.extend([
            cash / self.initial_capital,  # normalized cash
            current_position / self.max_position_size,  # normalized position
            (portfolio_value - self.initial_capital) / self.initial_capital,  # normalized PnL
            0.0,  # position duration (simplified)
            market_state.mid_price if market_state else 0.5,  # avg entry price (simplified)
            abs(current_position) / self.max_position_size,  # risk exposure
        ])

        # Game features (if enabled)
        if self.include_game_features:
            features.extend([
                self.game_state.quarter / 4.0,
                self.game_state.time_remaining / 3600.0,
                self.game_state.score_home / 50.0,  # normalized score
                self.game_state.score_away / 50.0,
                self.game_state.down / 4.0,
                self.game_state.distance / 100.0,
                self.game_state.field_position / 100.0,
                1.0 if self.game_state.possession == 'home' else 0.0,
                len(self.game_state.recent_events) / 10.0
            ])

        return np.array(features, dtype=np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics"""

        if self.feature_means is None:
            # Initialize with current features
            self.feature_means = features.copy()
            self.feature_stds = np.ones_like(features)
            return features

        # Update running mean and std (simplified)
        alpha = 0.01
        self.feature_means = (1 - alpha) * self.feature_means + alpha * features
        self.feature_stds = (1 - alpha) * self.feature_stds + alpha * np.abs(features - self.feature_means)

        # Avoid division by zero
        self.feature_stds = np.maximum(self.feature_stds, 1e-6)

        # Normalize
        normalized = (features - self.feature_means) / self.feature_stds

        # Clip extreme values
        normalized = np.clip(normalized, -5.0, 5.0)

        return normalized

    def _get_current_position_size(self) -> int:
        """Get current position size for the ticker"""
        positions = self.trading_env.get_position_summary()
        return positions.get(self.current_ticker, {}).get('size', 0)

    def _advance_time(self):
        """Advance time in the trading environment"""
        # This is a simplified time advancement - in reality this would be handled by the backtester
        self.trading_env.current_timestamp += 60  # Advance by 1 minute

    def _update_game_state(self):
        """Update simulated game state"""
        # Simplified game state progression
        self.game_state.time_remaining = max(0, self.game_state.time_remaining - 60)

        if self.game_state.time_remaining <= 0:
            self.game_state.quarter += 1
            self.game_state.time_remaining = 900 if self.game_state.quarter <= 4 else 0

        # Simulate occasional scoring events
        if np.random.random() < 0.01:  # 1% chance per minute
            if np.random.random() < 0.5:
                self.game_state.score_home += np.random.choice([3, 6, 7])
            else:
                self.game_state.score_away += np.random.choice([3, 6, 7])

    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End episode if game is over
        if self.game_state.quarter > 4:
            return True

        # End if no market data available
        if not self.trading_env.get_current_market_state(self.current_ticker):
            return True

        # End if severe loss
        portfolio_value = self.trading_env.get_portfolio_value()
        if portfolio_value < self.initial_capital * 0.5:  # 50% drawdown limit
            return True

        return False

    def _calculate_episode_metrics(self):
        """Calculate final episode performance metrics"""
        if not self.episode_rewards:
            return

        # Total return
        final_portfolio_value = self.trading_env.get_portfolio_value()
        self.episode_metrics['total_return'] = (final_portfolio_value - self.initial_capital) / self.initial_capital

        # Sharpe ratio (simplified)
        returns = np.array(self.episode_rewards)
        if np.std(returns) > 0:
            self.episode_metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Max drawdown (simplified)
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max)
        self.episode_metrics['max_drawdown'] = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

        # Win rate
        profitable_actions = np.sum(np.array(self.episode_rewards) > 0)
        self.episode_metrics['win_rate'] = profitable_actions / len(self.episode_rewards) if self.episode_rewards else 0.0

    def _create_mock_episode(self):
        """Create mock data for testing when real data is not available"""
        logger.info("Creating mock episode data")

        # Create mock market data
        self.current_ticker = "MOCK-TICKER"

        # Generate synthetic price series
        timestamps = np.arange(0, 3600, 60)  # 1 hour of data, 1-minute intervals
        initial_price = 0.5
        prices = [initial_price]

        for i in range(1, len(timestamps)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)
            new_price = max(0.01, min(0.99, prices[-1] + change))
            prices.append(new_price)

        # Store mock market data
        self.trading_env.market_data = {}
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            market_state = MarketState(
                timestamp=timestamp,
                ticker=self.current_ticker,
                bid_price=price - 0.01,
                ask_price=price + 0.01,
                mid_price=price,
                volume=1000 + np.random.randint(-200, 200),
                open_interest=50000
            )
            if self.current_ticker not in self.trading_env.market_data:
                self.trading_env.market_data[self.current_ticker] = []
            self.trading_env.market_data[self.current_ticker].append(market_state)

        logger.info(f"Created mock episode with {len(prices)} price points")

    def render(self, mode='human'):
        """Render environment (optional)"""
        if mode == 'human':
            portfolio_value = self.trading_env.get_portfolio_value()
            position = self._get_current_position_size()
            print(f"Portfolio Value: ${portfolio_value:.2f}, Position: {position}, Quarter: {self.game_state.quarter}")

    def close(self):
        """Clean up environment"""
        pass

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get detailed episode statistics"""
        return {
            'episode_length': len(self.episode_rewards),
            'total_reward': sum(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'final_portfolio_value': self.trading_env.get_portfolio_value(),
            'total_return': self.episode_metrics['total_return'],
            'sharpe_ratio': self.episode_metrics['sharpe_ratio'],
            'max_drawdown': self.episode_metrics['max_drawdown'],
            'transaction_costs': self.episode_metrics['transaction_costs'],
            'num_trades': self.episode_metrics['num_trades'],
            'win_rate': self.episode_metrics['win_rate']
        }