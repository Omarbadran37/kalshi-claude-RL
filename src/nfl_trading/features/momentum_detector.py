"""Momentum detection for NFL trading system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..config import get_config, get_logger


logger = get_logger(__name__)


class MomentumType(Enum):
    """Types of momentum detected."""
    GAME_MOMENTUM = "game_momentum"
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_MOMENTUM = "volume_momentum"
    COMBINED_MOMENTUM = "combined_momentum"


class MomentumDirection(Enum):
    """Direction of momentum."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EventType(Enum):
    """Types of NFL events that affect momentum."""
    TOUCHDOWN = "touchdown"
    FIELD_GOAL = "field_goal"
    TURNOVER = "turnover"
    BIG_PLAY = "big_play"
    PENALTY = "penalty"
    TIMEOUT = "timeout"
    INJURY = "injury"
    RED_ZONE_ENTRY = "red_zone_entry"
    FOURTH_DOWN_CONVERSION = "fourth_down_conversion"
    SACK = "sack"


@dataclass
class MomentumEvent:
    """Represents a momentum shift event."""
    timestamp: datetime
    event_type: EventType
    momentum_type: MomentumType
    direction: MomentumDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    duration_estimate: timedelta
    description: str
    game_context: Optional[Dict[str, Any]] = None
    price_context: Optional[Dict[str, Any]] = None


@dataclass
class MomentumState:
    """Current momentum state."""
    current_strength: float
    direction: MomentumDirection
    duration: timedelta
    last_update: datetime
    contributing_factors: List[str]


class MomentumDetector:
    """Detects momentum shifts combining NFL game events and price movements."""

    def __init__(self, config=None):
        """Initialize momentum detector.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.MomentumDetector")

        # Detection thresholds
        self.price_momentum_threshold = 0.02  # 2% price change
        self.volume_spike_threshold = 2.0     # 2x average volume
        self.big_play_threshold = 20          # Yards for big play
        self.momentum_decay_period = timedelta(minutes=5)
        self.breakout_threshold = 0.03        # 3% breakout from range
        
        # Event impact weights
        self.event_weights = {
            EventType.TOUCHDOWN: 0.9,
            EventType.TURNOVER: 0.8,
            EventType.BIG_PLAY: 0.6,
            EventType.FIELD_GOAL: 0.4,
            EventType.FOURTH_DOWN_CONVERSION: 0.5,
            EventType.SACK: 0.3,
            EventType.RED_ZONE_ENTRY: 0.4,
            EventType.PENALTY: 0.2,
            EventType.TIMEOUT: 0.1,
            EventType.INJURY: 0.3
        }
        
        # Current state tracking
        self.momentum_state = MomentumState(
            current_strength=0.0,
            direction=MomentumDirection.NEUTRAL,
            duration=timedelta(0),
            last_update=datetime.now(),
            contributing_factors=[]
        )

    def detect_momentum(self, aligned_data: pd.DataFrame, 
                       team_focus: Optional[str] = None) -> Tuple[pd.DataFrame, List[MomentumEvent]]:
        """Detect momentum shifts from aligned NFL and price data.

        Args:
            aligned_data: DataFrame with aligned NFL plays and price data
            team_focus: Team to focus momentum analysis on

        Returns:
            Tuple of (features_df, momentum_events)
        """
        try:
            self.logger.info(f"Detecting momentum from {len(aligned_data)} aligned records")
            
            # Validate input data
            aligned_data = self._validate_aligned_data(aligned_data)
            
            # Initialize features dataframe
            features_df = aligned_data.copy()
            momentum_events = []
            
            # Detect game momentum
            features_df, game_events = self._detect_game_momentum(features_df, team_focus)
            momentum_events.extend(game_events)
            
            # Detect price momentum
            features_df, price_events = self._detect_price_momentum(features_df)
            momentum_events.extend(price_events)
            
            # Detect volume momentum
            features_df, volume_events = self._detect_volume_momentum(features_df)
            momentum_events.extend(volume_events)
            
            # Detect combined momentum patterns
            features_df, combined_events = self._detect_combined_momentum(features_df)
            momentum_events.extend(combined_events)
            
            # Add momentum state features
            features_df = self._add_momentum_state_features(features_df, momentum_events)
            
            # Detect mean reversion signals
            features_df = self._detect_mean_reversion(features_df)
            
            # Sort events by timestamp
            momentum_events.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Detected {len(momentum_events)} momentum events")
            return features_df, momentum_events
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum: {e}")
            raise

    def _validate_aligned_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate aligned data has required columns."""
        required_cols = ['timestamp', 'play_type', 'yards_gained', 'close_price', 'volume']
        optional_cols = ['score_differential', 'field_position', 'quarter', 'time_remaining']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Fill missing optional columns
        for col in optional_cols:
            if col not in data.columns:
                data[col] = 0
        
        return data

    def _detect_game_momentum(self, df: pd.DataFrame, 
                            team_focus: Optional[str] = None) -> Tuple[pd.DataFrame, List[MomentumEvent]]:
        """Detect momentum shifts from game events."""
        events = []
        
        for idx, row in df.iterrows():
            # Identify significant game events
            game_events = self._identify_game_events(row)
            
            # Calculate momentum features
            momentum_features = self._calculate_game_momentum_features(df, idx, team_focus)
            
            # Add features to dataframe
            for key, value in momentum_features.items():
                df.loc[idx, f'game_momentum_{key}'] = value
            
            # Create momentum events
            for event_type in game_events:
                strength = self._calculate_event_strength(event_type, row, df, idx)
                direction = self._determine_momentum_direction(event_type, row, team_focus)
                
                if strength > 0.3:  # Only significant events
                    event = MomentumEvent(
                        timestamp=row['timestamp'],
                        event_type=event_type,
                        momentum_type=MomentumType.GAME_MOMENTUM,
                        direction=direction,
                        strength=strength,
                        confidence=self._calculate_confidence(event_type, row),
                        duration_estimate=self._estimate_duration(event_type, strength),
                        description=self._create_event_description(event_type, row),
                        game_context=self._extract_game_context(row),
                        price_context=None
                    )
                    events.append(event)
        
        return df, events

    def _identify_game_events(self, row: pd.Series) -> List[EventType]:
        """Identify significant game events from a play."""
        events = []
        
        play_type = str(row.get('play_type', '')).lower()
        yards_gained = row.get('yards_gained', 0)
        
        # Scoring events
        if 'touchdown' in play_type:
            events.append(EventType.TOUCHDOWN)
        elif 'field_goal' in play_type:
            events.append(EventType.FIELD_GOAL)
        
        # Turnovers
        if any(keyword in play_type for keyword in ['interception', 'fumble', 'turnover']):
            events.append(EventType.TURNOVER)
        
        # Big plays
        if yards_gained >= self.big_play_threshold:
            events.append(EventType.BIG_PLAY)
        
        # Fourth down conversions
        if row.get('down') == 4 and yards_gained >= row.get('distance', 0):
            events.append(EventType.FOURTH_DOWN_CONVERSION)
        
        # Sacks
        if 'sack' in play_type:
            events.append(EventType.SACK)
        
        # Red zone entry
        if row.get('field_position', 50) >= 80:
            events.append(EventType.RED_ZONE_ENTRY)
        
        # Penalties
        if 'penalty' in play_type:
            events.append(EventType.PENALTY)
        
        return events

    def _calculate_game_momentum_features(self, df: pd.DataFrame, current_idx: int,
                                        team_focus: Optional[str] = None) -> Dict[str, float]:
        """Calculate game momentum features."""
        features = {}
        
        # Look at recent plays (last 5 plays)
        start_idx = max(0, current_idx - 5)
        recent_plays = df.iloc[start_idx:current_idx+1]
        
        if len(recent_plays) == 0:
            return {key: 0.0 for key in ['score_trend', 'yards_trend', 'success_rate', 
                                       'big_play_momentum', 'field_position_trend']}
        
        # Score momentum
        if len(recent_plays) > 1:
            first_score = recent_plays.iloc[0].get('score_differential', 0)
            last_score = recent_plays.iloc[-1].get('score_differential', 0)
            features['score_trend'] = last_score - first_score
        else:
            features['score_trend'] = 0.0
        
        # Yards gained momentum
        yards = recent_plays['yards_gained'].fillna(0)
        features['yards_trend'] = yards.sum()
        features['avg_yards_per_play'] = yards.mean()
        
        # Success rate (positive yard plays)
        successful_plays = (yards > 0).sum()
        features['success_rate'] = successful_plays / len(recent_plays)
        
        # Big play momentum
        big_plays = (yards >= self.big_play_threshold).sum()
        features['big_play_momentum'] = big_plays / len(recent_plays)
        
        # Field position trend
        if len(recent_plays) > 1:
            first_pos = recent_plays.iloc[0].get('field_position', 50)
            last_pos = recent_plays.iloc[-1].get('field_position', 50)
            features['field_position_trend'] = last_pos - first_pos
        else:
            features['field_position_trend'] = 0.0
        
        # Time pressure factor
        current_row = df.iloc[current_idx]
        quarter = current_row.get('quarter', 1)
        time_remaining = current_row.get('time_remaining', 3600)
        
        if quarter >= 4 and time_remaining <= 300:  # Last 5 minutes
            features['time_pressure'] = 1.0
        elif quarter >= 4 and time_remaining <= 600:  # Last 10 minutes
            features['time_pressure'] = 0.5
        else:
            features['time_pressure'] = 0.0
        
        return features

    def _detect_price_momentum(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[MomentumEvent]]:
        """Detect momentum from price movements."""
        events = []
        
        # Calculate price momentum indicators
        df['price_change_1'] = df['close_price'].pct_change()
        df['price_change_3'] = df['close_price'].pct_change(periods=3)
        df['price_change_5'] = df['close_price'].pct_change(periods=5)
        
        # Moving averages for trend detection
        df['price_sma_5'] = df['close_price'].rolling(5).mean()
        df['price_sma_10'] = df['close_price'].rolling(10).mean()
        
        # Price momentum features
        df['price_momentum_strength'] = abs(df['price_change_5'])
        df['price_momentum_direction'] = np.sign(df['price_change_5'])
        df['price_above_sma'] = (df['close_price'] > df['price_sma_5']).astype(int)
        
        # Detect breakouts
        df['price_breakout'] = self._detect_price_breakouts(df)
        
        # Identify significant price movements
        for idx, row in df.iterrows():
            if abs(row['price_change_1']) > self.price_momentum_threshold:
                direction = MomentumDirection.BULLISH if row['price_change_1'] > 0 else MomentumDirection.BEARISH
                strength = min(abs(row['price_change_1']) / self.price_momentum_threshold, 1.0)
                
                event = MomentumEvent(
                    timestamp=row['timestamp'],
                    event_type=EventType.BIG_PLAY,  # Using as generic significant event
                    momentum_type=MomentumType.PRICE_MOMENTUM,
                    direction=direction,
                    strength=strength,
                    confidence=0.7,
                    duration_estimate=timedelta(minutes=2),
                    description=f"Price movement: {row['price_change_1']:.2%}",
                    game_context=None,
                    price_context={'price_change': row['price_change_1'], 'price': row['close_price']}
                )
                events.append(event)
        
        return df, events

    def _detect_volume_momentum(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[MomentumEvent]]:
        """Detect momentum from volume patterns."""
        events = []
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_spike'] = (df['volume_ratio'] > self.volume_spike_threshold).astype(int)
        
        # Volume momentum features
        df['volume_momentum'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].rolling(5).sum() - df['volume'].rolling(5, offset=5).sum()
        
        # Detect volume spikes
        for idx, row in df.iterrows():
            if row['volume_spike'] == 1:
                # Determine direction based on concurrent price movement
                price_change = row.get('price_change_1', 0)
                direction = MomentumDirection.BULLISH if price_change > 0 else MomentumDirection.BEARISH
                if abs(price_change) < 0.001:
                    direction = MomentumDirection.NEUTRAL
                
                strength = min(row['volume_ratio'] / self.volume_spike_threshold, 1.0)
                
                event = MomentumEvent(
                    timestamp=row['timestamp'],
                    event_type=EventType.BIG_PLAY,
                    momentum_type=MomentumType.VOLUME_MOMENTUM,
                    direction=direction,
                    strength=strength,
                    confidence=0.6,
                    duration_estimate=timedelta(minutes=3),
                    description=f"Volume spike: {row['volume_ratio']:.1f}x average",
                    game_context=None,
                    price_context={'volume_ratio': row['volume_ratio'], 'volume': row['volume']}
                )
                events.append(event)
        
        return df, events

    def _detect_combined_momentum(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[MomentumEvent]]:
        """Detect combined momentum from game events and price movements."""
        events = []
        
        # Combined momentum score
        df['combined_momentum'] = (
            df.get('game_momentum_success_rate', 0) * 0.3 +
            df.get('price_momentum_strength', 0) * 0.4 +
            df.get('volume_momentum', 1) * 0.3
        )
        
        # Momentum alignment score
        game_direction = df.get('game_momentum_direction', 0)
        price_direction = df.get('price_momentum_direction', 0)
        df['momentum_alignment'] = abs(game_direction - price_direction)
        
        # Detect significant combined momentum
        for idx, row in df.iterrows():
            combined_score = row.get('combined_momentum', 0)
            alignment = row.get('momentum_alignment', 0)
            
            if combined_score > 0.6 and alignment < 0.5:  # Strong aligned momentum
                direction = MomentumDirection.BULLISH if combined_score > 0 else MomentumDirection.BEARISH
                
                event = MomentumEvent(
                    timestamp=row['timestamp'],
                    event_type=EventType.BIG_PLAY,
                    momentum_type=MomentumType.COMBINED_MOMENTUM,
                    direction=direction,
                    strength=combined_score,
                    confidence=0.8,
                    duration_estimate=timedelta(minutes=5),
                    description=f"Combined momentum: {combined_score:.2f}",
                    game_context=self._extract_game_context(row),
                    price_context={'combined_score': combined_score, 'alignment': alignment}
                )
                events.append(event)
        
        return df, events

    def _detect_price_breakouts(self, df: pd.DataFrame) -> pd.Series:
        """Detect price breakout patterns."""
        breakouts = pd.Series(0, index=df.index)
        
        # Calculate recent high/low ranges
        df['recent_high'] = df['close_price'].rolling(20).max()
        df['recent_low'] = df['close_price'].rolling(20).min()
        df['price_range'] = df['recent_high'] - df['recent_low']
        
        # Detect breakouts above recent high
        upward_breakout = (
            (df['close_price'] > df['recent_high'] * (1 + self.breakout_threshold)) &
            (df['price_range'] > 0)
        )
        
        # Detect breakdowns below recent low
        downward_breakout = (
            (df['close_price'] < df['recent_low'] * (1 - self.breakout_threshold)) &
            (df['price_range'] > 0)
        )
        
        breakouts[upward_breakout] = 1
        breakouts[downward_breakout] = -1
        
        return breakouts

    def _detect_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect mean reversion signals."""
        # Calculate deviation from moving average
        df['price_deviation'] = (df['close_price'] / df['price_sma_10'] - 1) * 100
        
        # Mean reversion signals
        df['mean_reversion_signal'] = 0
        
        # Oversold (potential reversal up)
        oversold = df['price_deviation'] < -5  # 5% below moving average
        df.loc[oversold, 'mean_reversion_signal'] = 1
        
        # Overbought (potential reversal down)
        overbought = df['price_deviation'] > 5  # 5% above moving average
        df.loc[overbought, 'mean_reversion_signal'] = -1
        
        # Reversion strength
        df['reversion_strength'] = abs(df['price_deviation']) / 10  # Normalize to 0-1
        df['reversion_strength'] = np.clip(df['reversion_strength'], 0, 1)
        
        return df

    def _add_momentum_state_features(self, df: pd.DataFrame, 
                                   events: List[MomentumEvent]) -> pd.DataFrame:
        """Add momentum state tracking features."""
        # Initialize momentum state columns
        df['momentum_state_strength'] = 0.0
        df['momentum_state_direction'] = 0
        df['momentum_events_recent'] = 0
        
        current_momentum = 0.0
        momentum_direction = 0
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            
            # Find recent events (within last 5 minutes)
            recent_events = [e for e in events 
                           if e.timestamp <= timestamp and 
                              (timestamp - e.timestamp) <= self.momentum_decay_period]
            
            # Calculate current momentum state
            if recent_events:
                total_strength = sum(e.strength for e in recent_events)
                avg_direction = np.mean([1 if e.direction == MomentumDirection.BULLISH else -1 
                                       for e in recent_events])
                
                # Apply time decay
                weighted_strength = 0
                for event in recent_events:
                    time_diff = (timestamp - event.timestamp).total_seconds()
                    decay_factor = np.exp(-time_diff / self.momentum_decay_period.total_seconds())
                    weighted_strength += event.strength * decay_factor
                
                current_momentum = weighted_strength
                momentum_direction = avg_direction
            else:
                # Decay momentum over time
                current_momentum *= 0.95
                if abs(current_momentum) < 0.1:
                    current_momentum = 0.0
                    momentum_direction = 0
            
            df.loc[idx, 'momentum_state_strength'] = current_momentum
            df.loc[idx, 'momentum_state_direction'] = momentum_direction
            df.loc[idx, 'momentum_events_recent'] = len(recent_events)
        
        return df

    def _calculate_event_strength(self, event_type: EventType, row: pd.Series,
                                df: pd.DataFrame, idx: int) -> float:
        """Calculate the strength of a game event."""
        base_strength = self.event_weights.get(event_type, 0.3)
        
        # Context multipliers
        multiplier = 1.0
        
        # Game situation multiplier
        score_diff = abs(row.get('score_differential', 0))
        if score_diff <= 7:  # Close game
            multiplier *= 1.5
        elif score_diff <= 14:  # Moderate lead
            multiplier *= 1.2
        
        # Time pressure multiplier
        quarter = row.get('quarter', 1)
        time_remaining = row.get('time_remaining', 3600)
        
        if quarter >= 4 and time_remaining <= 300:  # Last 5 minutes
            multiplier *= 2.0
        elif quarter >= 4 and time_remaining <= 600:  # Last 10 minutes
            multiplier *= 1.5
        
        # Field position multiplier (for offensive events)
        if event_type in [EventType.TOUCHDOWN, EventType.BIG_PLAY]:
            field_pos = row.get('field_position', 50)
            if field_pos >= 80:  # Red zone
                multiplier *= 1.3
        
        return min(base_strength * multiplier, 1.0)

    def _determine_momentum_direction(self, event_type: EventType, row: pd.Series,
                                    team_focus: Optional[str] = None) -> MomentumDirection:
        """Determine momentum direction for an event."""
        # Positive events
        positive_events = [EventType.TOUCHDOWN, EventType.FIELD_GOAL, EventType.BIG_PLAY,
                          EventType.FOURTH_DOWN_CONVERSION, EventType.RED_ZONE_ENTRY]
        
        # Negative events
        negative_events = [EventType.TURNOVER, EventType.SACK, EventType.PENALTY]
        
        if event_type in positive_events:
            return MomentumDirection.BULLISH
        elif event_type in negative_events:
            return MomentumDirection.BEARISH
        else:
            return MomentumDirection.NEUTRAL

    def _calculate_confidence(self, event_type: EventType, row: pd.Series) -> float:
        """Calculate confidence in the momentum signal."""
        base_confidence = {
            EventType.TOUCHDOWN: 0.9,
            EventType.TURNOVER: 0.8,
            EventType.BIG_PLAY: 0.7,
            EventType.FIELD_GOAL: 0.6,
            EventType.FOURTH_DOWN_CONVERSION: 0.7,
            EventType.SACK: 0.6,
            EventType.RED_ZONE_ENTRY: 0.5,
            EventType.PENALTY: 0.4,
            EventType.TIMEOUT: 0.3,
            EventType.INJURY: 0.5
        }.get(event_type, 0.5)
        
        # Adjust based on game context
        quarter = row.get('quarter', 1)
        if quarter >= 4:
            base_confidence *= 1.2
        
        return min(base_confidence, 1.0)

    def _estimate_duration(self, event_type: EventType, strength: float) -> timedelta:
        """Estimate how long momentum from this event will last."""
        base_durations = {
            EventType.TOUCHDOWN: timedelta(minutes=10),
            EventType.TURNOVER: timedelta(minutes=8),
            EventType.BIG_PLAY: timedelta(minutes=5),
            EventType.FIELD_GOAL: timedelta(minutes=3),
            EventType.FOURTH_DOWN_CONVERSION: timedelta(minutes=4),
            EventType.SACK: timedelta(minutes=3),
            EventType.RED_ZONE_ENTRY: timedelta(minutes=2),
            EventType.PENALTY: timedelta(minutes=1),
            EventType.TIMEOUT: timedelta(minutes=1),
            EventType.INJURY: timedelta(minutes=5)
        }.get(event_type, timedelta(minutes=2))
        
        # Scale by strength
        return base_durations * (0.5 + strength * 0.5)

    def _create_event_description(self, event_type: EventType, row: pd.Series) -> str:
        """Create a human-readable description of the event."""
        descriptions = {
            EventType.TOUCHDOWN: f"Touchdown - {row.get('yards_gained', 0)} yard score",
            EventType.FIELD_GOAL: "Field goal scored",
            EventType.TURNOVER: "Turnover occurred",
            EventType.BIG_PLAY: f"Big play - {row.get('yards_gained', 0)} yards gained",
            EventType.FOURTH_DOWN_CONVERSION: "Fourth down converted",
            EventType.SACK: "Quarterback sacked",
            EventType.RED_ZONE_ENTRY: "Entered red zone",
            EventType.PENALTY: "Penalty called",
            EventType.TIMEOUT: "Timeout called",
            EventType.INJURY: "Player injury"
        }
        
        return descriptions.get(event_type, f"{event_type.value} event")

    def _extract_game_context(self, row: pd.Series) -> Dict[str, Any]:
        """Extract relevant game context."""
        return {
            'quarter': row.get('quarter', 1),
            'time_remaining': row.get('time_remaining', 3600),
            'score_differential': row.get('score_differential', 0),
            'field_position': row.get('field_position', 50),
            'down': row.get('down', 1),
            'distance': row.get('distance', 10),
            'play_type': row.get('play_type', 'unknown')
        }

    def get_momentum_summary(self, events: List[MomentumEvent]) -> Dict[str, Any]:
        """Generate a summary of detected momentum events."""
        if not events:
            return {
                'total_events': 0,
                'momentum_types': {},
                'directions': {},
                'avg_strength': 0.0,
                'strongest_event': None
            }
        
        # Count by type
        momentum_types = {}
        for event in events:
            momentum_types[event.momentum_type.value] = momentum_types.get(event.momentum_type.value, 0) + 1
        
        # Count by direction
        directions = {}
        for event in events:
            directions[event.direction.value] = directions.get(event.direction.value, 0) + 1
        
        # Find strongest event
        strongest_event = max(events, key=lambda x: x.strength)
        
        return {
            'total_events': len(events),
            'momentum_types': momentum_types,
            'directions': directions,
            'avg_strength': np.mean([e.strength for e in events]),
            'avg_confidence': np.mean([e.confidence for e in events]),
            'strongest_event': {
                'timestamp': strongest_event.timestamp,
                'type': strongest_event.event_type.value,
                'strength': strongest_event.strength,
                'description': strongest_event.description
            }
        }

    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of momentum feature names to descriptions."""
        return {
            # Game momentum features
            'game_momentum_score_trend': 'Score differential trend in recent plays',
            'game_momentum_yards_trend': 'Total yards gained in recent plays',
            'game_momentum_success_rate': 'Success rate of recent plays',
            'game_momentum_big_play_momentum': 'Rate of big plays in recent plays',
            'game_momentum_field_position_trend': 'Field position improvement trend',
            'game_momentum_time_pressure': 'Time pressure factor (0-1)',
            
            # Price momentum features
            'price_change_1': '1-period price change',
            'price_change_3': '3-period price change',
            'price_change_5': '5-period price change',
            'price_momentum_strength': 'Absolute strength of price momentum',
            'price_momentum_direction': 'Direction of price momentum (-1, 0, 1)',
            'price_above_sma': 'Price above 5-period moving average (1/0)',
            'price_breakout': 'Price breakout indicator (-1, 0, 1)',
            
            # Volume momentum features
            'volume_ratio': 'Volume ratio vs 20-period average',
            'volume_spike': 'Volume spike indicator (1/0)',
            'volume_momentum': 'Volume momentum (5-period vs 20-period)',
            'volume_trend': 'Volume trend in recent periods',
            
            # Combined momentum features
            'combined_momentum': 'Combined momentum score (0-1)',
            'momentum_alignment': 'Alignment between game and price momentum',
            
            # Momentum state features
            'momentum_state_strength': 'Current momentum state strength',
            'momentum_state_direction': 'Current momentum state direction',
            'momentum_events_recent': 'Number of recent momentum events',
            
            # Mean reversion features
            'price_deviation': 'Price deviation from 10-period SMA (%)',
            'mean_reversion_signal': 'Mean reversion signal (-1, 0, 1)',
            'reversion_strength': 'Strength of mean reversion signal (0-1)'
        }