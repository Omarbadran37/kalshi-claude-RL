"""Tests for MomentumDetector class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.nfl_trading.features.momentum_detector import (
    MomentumDetector, MomentumEvent, MomentumType, MomentumDirection, EventType
)


class TestMomentumDetector:
    """Test cases for MomentumDetector."""

    @pytest.fixture
    def detector(self):
        """Create MomentumDetector instance."""
        return MomentumDetector()

    @pytest.fixture
    def sample_aligned_data(self):
        """Create sample aligned NFL and price data."""
        base_time = datetime.now()
        n_points = 50
        
        # Generate correlated price and game data
        np.random.seed(42)
        prices = 50 + np.cumsum(np.random.randn(n_points) * 0.02)
        volumes = np.random.randint(1000, 5000, n_points)
        
        data = {
            'timestamp': [base_time + timedelta(minutes=i*2) for i in range(n_points)],
            'play_type': ['pass', 'run', 'touchdown', 'punt', 'field_goal'] * 10,
            'yards_gained': [5, 12, 25, -2, 0] * 10,
            'close_price': prices,
            'volume': volumes,
            'score_differential': [0, 0, 7, 7, 10] * 10,
            'field_position': [25, 35, 85, 45, 75] * 10,
            'quarter': [1, 1, 1, 2, 2] * 10,
            'time_remaining': [3600 - i*60 for i in range(n_points)],
            'down': [1, 2, 1, 4, 1] * 10,
            'distance': [10, 7, 10, 8, 10] * 10,
            'possession_team': ['DAL', 'DAL', 'DAL', 'PHI', 'PHI'] * 10,
            'home_team': ['DAL'] * n_points,
            'away_team': ['PHI'] * n_points
        }
        
        return pd.DataFrame(data)

    def test_initialization(self, detector):
        """Test MomentumDetector initialization."""
        assert detector.price_momentum_threshold == 0.02
        assert detector.volume_spike_threshold == 2.0
        assert detector.big_play_threshold == 20
        assert detector.breakout_threshold == 0.03
        
        # Check event weights
        assert EventType.TOUCHDOWN in detector.event_weights
        assert detector.event_weights[EventType.TOUCHDOWN] == 0.9

    def test_detect_momentum_basic(self, detector, sample_aligned_data):
        """Test basic momentum detection."""
        features_df, momentum_events = detector.detect_momentum(sample_aligned_data)
        
        # Should return dataframe and events
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(momentum_events, list)
        assert len(features_df) == len(sample_aligned_data)
        
        # Check momentum events
        if momentum_events:
            event = momentum_events[0]
            assert isinstance(event, MomentumEvent)
            assert hasattr(event, 'timestamp')
            assert hasattr(event, 'strength')
            assert hasattr(event, 'confidence')

    def test_validate_aligned_data(self, detector):
        """Test aligned data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'yards_gained': [5],
            'close_price': [50.0],
            'volume': [1000]
        })
        
        validated = detector._validate_aligned_data(valid_data)
        assert len(validated) == 1
        
        # Missing required columns
        invalid_data = pd.DataFrame({'timestamp': [datetime.now()]})
        with pytest.raises(ValueError, match="Missing required columns"):
            detector._validate_aligned_data(invalid_data)

    def test_identify_game_events(self, detector, sample_aligned_data):
        """Test game event identification."""
        # Test touchdown detection
        touchdown_row = pd.Series({
            'play_type': 'touchdown pass',
            'yards_gained': 25,
            'down': 1,
            'distance': 10,
            'field_position': 85
        })
        
        events = detector._identify_game_events(touchdown_row)
        assert EventType.TOUCHDOWN in events
        assert EventType.BIG_PLAY in events  # 25 yards >= 20
        assert EventType.RED_ZONE_ENTRY in events  # field_position >= 80
        
        # Test turnover detection
        turnover_row = pd.Series({
            'play_type': 'interception',
            'yards_gained': -5,
            'down': 3,
            'distance': 5,
            'field_position': 50
        })
        
        events = detector._identify_game_events(turnover_row)
        assert EventType.TURNOVER in events
        
        # Test field goal detection
        fg_row = pd.Series({
            'play_type': 'field goal good',
            'yards_gained': 0,
            'down': 4,
            'distance': 3,
            'field_position': 70
        })
        
        events = detector._identify_game_events(fg_row)
        assert EventType.FIELD_GOAL in events

    def test_game_momentum_features(self, detector, sample_aligned_data):
        """Test game momentum feature calculation."""
        features_df, _ = detector._detect_game_momentum(sample_aligned_data)
        
        # Check game momentum features
        momentum_cols = [col for col in features_df.columns if 'game_momentum_' in col]
        assert len(momentum_cols) > 0
        
        expected_features = [
            'game_momentum_score_trend',
            'game_momentum_yards_trend',
            'game_momentum_success_rate',
            'game_momentum_big_play_momentum',
            'game_momentum_field_position_trend'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns

    def test_price_momentum_detection(self, detector, sample_aligned_data):
        """Test price momentum detection."""
        # Add significant price changes
        data = sample_aligned_data.copy()
        data.loc[5, 'close_price'] = data.loc[4, 'close_price'] * 1.05  # 5% increase
        data.loc[10, 'close_price'] = data.loc[9, 'close_price'] * 0.95  # 5% decrease
        
        features_df, price_events = detector._detect_price_momentum(data)
        
        # Check price momentum features
        assert 'price_change_1' in features_df.columns
        assert 'price_change_3' in features_df.columns
        assert 'price_momentum_strength' in features_df.columns
        assert 'price_momentum_direction' in features_df.columns
        
        # Should detect significant price movements
        price_momentum_events = [e for e in price_events if e.momentum_type == MomentumType.PRICE_MOMENTUM]
        assert len(price_momentum_events) > 0

    def test_volume_momentum_detection(self, detector, sample_aligned_data):
        """Test volume momentum detection."""
        # Add volume spikes
        data = sample_aligned_data.copy()
        avg_volume = data['volume'].mean()
        data.loc[5, 'volume'] = avg_volume * 3  # 3x spike
        data.loc[10, 'volume'] = avg_volume * 2.5  # 2.5x spike
        
        features_df, volume_events = detector._detect_volume_momentum(data)
        
        # Check volume momentum features
        assert 'volume_sma_20' in features_df.columns
        assert 'volume_ratio' in features_df.columns
        assert 'volume_spike' in features_df.columns
        assert 'volume_momentum' in features_df.columns
        
        # Should detect volume spikes
        volume_momentum_events = [e for e in volume_events if e.momentum_type == MomentumType.VOLUME_MOMENTUM]
        assert len(volume_momentum_events) > 0

    def test_combined_momentum_detection(self, detector, sample_aligned_data):
        """Test combined momentum detection."""
        features_df, combined_events = detector._detect_combined_momentum(sample_aligned_data)
        
        # Check combined momentum features
        assert 'combined_momentum' in features_df.columns
        assert 'momentum_alignment' in features_df.columns
        
        # Combined momentum should be calculated
        assert not features_df['combined_momentum'].isnull().all()

    def test_price_breakout_detection(self, detector, sample_aligned_data):
        """Test price breakout pattern detection."""
        # Create breakout pattern
        data = sample_aligned_data.copy()
        base_price = data['close_price'].iloc[0]
        
        # Establish range
        data.loc[0:10, 'close_price'] = base_price + np.random.randn(11) * 0.01
        
        # Create breakout
        data.loc[15, 'close_price'] = base_price * 1.05  # 5% breakout
        
        breakouts = detector._detect_price_breakouts(data)
        
        # Should detect breakout
        assert (breakouts != 0).any()

    def test_mean_reversion_detection(self, detector, sample_aligned_data):
        """Test mean reversion signal detection."""
        features_df = detector._detect_mean_reversion(sample_aligned_data)
        
        # Check mean reversion features
        assert 'price_deviation' in features_df.columns
        assert 'mean_reversion_signal' in features_df.columns
        assert 'reversion_strength' in features_df.columns
        
        # Mean reversion signals should be in valid range
        signals = features_df['mean_reversion_signal'].dropna()
        assert signals.isin([-1, 0, 1]).all()

    def test_momentum_state_features(self, detector, sample_aligned_data):
        """Test momentum state tracking."""
        # Create some momentum events
        events = [
            MomentumEvent(
                timestamp=sample_aligned_data.iloc[5]['timestamp'],
                event_type=EventType.TOUCHDOWN,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BULLISH,
                strength=0.8,
                confidence=0.9,
                duration_estimate=timedelta(minutes=5),
                description="Touchdown scored"
            ),
            MomentumEvent(
                timestamp=sample_aligned_data.iloc[10]['timestamp'],
                event_type=EventType.TURNOVER,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BEARISH,
                strength=0.7,
                confidence=0.8,
                duration_estimate=timedelta(minutes=3),
                description="Turnover occurred"
            )
        ]
        
        features_df = detector._add_momentum_state_features(sample_aligned_data, events)
        
        # Check momentum state features
        assert 'momentum_state_strength' in features_df.columns
        assert 'momentum_state_direction' in features_df.columns
        assert 'momentum_events_recent' in features_df.columns
        
        # State features should reflect events
        assert features_df['momentum_events_recent'].max() > 0

    def test_event_strength_calculation(self, detector, sample_aligned_data):
        """Test event strength calculation with context."""
        row = sample_aligned_data.iloc[0]
        
        # Test touchdown in close game
        row_close_game = row.copy()
        row_close_game['score_differential'] = 3  # Close game
        row_close_game['quarter'] = 4
        row_close_game['time_remaining'] = 120  # 2 minutes left
        
        strength = detector._calculate_event_strength(
            EventType.TOUCHDOWN, row_close_game, sample_aligned_data, 0
        )
        
        # Should have high strength due to context
        assert strength > 0.5
        
        # Test same event in blowout
        row_blowout = row.copy()
        row_blowout['score_differential'] = 28  # Blowout
        row_blowout['quarter'] = 1
        row_blowout['time_remaining'] = 3600
        
        strength_blowout = detector._calculate_event_strength(
            EventType.TOUCHDOWN, row_blowout, sample_aligned_data, 0
        )
        
        # Should have lower strength in blowout
        assert strength_blowout < strength

    def test_momentum_direction_determination(self, detector):
        """Test momentum direction determination."""
        # Test positive events
        play = pd.Series({'yards_gained': 25})
        
        direction = detector._determine_momentum_direction(EventType.TOUCHDOWN, play)
        assert direction == MomentumDirection.BULLISH
        
        direction = detector._determine_momentum_direction(EventType.BIG_PLAY, play)
        assert direction == MomentumDirection.BULLISH
        
        # Test negative events
        direction = detector._determine_momentum_direction(EventType.TURNOVER, play)
        assert direction == MomentumDirection.BEARISH
        
        direction = detector._determine_momentum_direction(EventType.SACK, play)
        assert direction == MomentumDirection.BEARISH
        
        # Test neutral events
        direction = detector._determine_momentum_direction(EventType.TIMEOUT, play)
        assert direction == MomentumDirection.NEUTRAL

    def test_confidence_calculation(self, detector, sample_aligned_data):
        """Test confidence calculation for events."""
        row = sample_aligned_data.iloc[0]
        
        # Touchdown should have high confidence
        confidence_td = detector._calculate_confidence(EventType.TOUCHDOWN, row)
        assert confidence_td > 0.8
        
        # Timeout should have lower confidence
        confidence_timeout = detector._calculate_confidence(EventType.TIMEOUT, row)
        assert confidence_timeout < confidence_td
        
        # Fourth quarter should increase confidence
        row_q4 = row.copy()
        row_q4['quarter'] = 4
        confidence_q4 = detector._calculate_confidence(EventType.TOUCHDOWN, row_q4)
        assert confidence_q4 >= confidence_td

    def test_duration_estimation(self, detector):
        """Test momentum duration estimation."""
        # High strength should last longer
        duration_high = detector._estimate_duration(EventType.TOUCHDOWN, 0.9)
        duration_low = detector._estimate_duration(EventType.TOUCHDOWN, 0.3)
        
        assert duration_high > duration_low
        
        # Different event types should have different base durations
        duration_td = detector._estimate_duration(EventType.TOUCHDOWN, 0.5)
        duration_timeout = detector._estimate_duration(EventType.TIMEOUT, 0.5)
        
        assert duration_td > duration_timeout

    def test_momentum_summary(self, detector):
        """Test momentum event summary generation."""
        events = [
            MomentumEvent(
                timestamp=datetime.now(),
                event_type=EventType.TOUCHDOWN,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BULLISH,
                strength=0.8,
                confidence=0.9,
                duration_estimate=timedelta(minutes=5),
                description="Touchdown"
            ),
            MomentumEvent(
                timestamp=datetime.now() + timedelta(minutes=5),
                event_type=EventType.TURNOVER,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BEARISH,
                strength=0.7,
                confidence=0.8,
                duration_estimate=timedelta(minutes=3),
                description="Turnover"
            )
        ]
        
        summary = detector.get_momentum_summary(events)
        
        assert summary['total_events'] == 2
        assert 'momentum_types' in summary
        assert 'directions' in summary
        assert 'avg_strength' in summary
        assert 'strongest_event' in summary
        
        # Check strongest event
        strongest = summary['strongest_event']
        assert strongest['strength'] == 0.8  # Touchdown was stronger

    def test_empty_events_summary(self, detector):
        """Test summary with no events."""
        summary = detector.get_momentum_summary([])
        
        assert summary['total_events'] == 0
        assert summary['avg_strength'] == 0.0
        assert summary['strongest_event'] is None

    def test_team_focus_functionality(self, detector, sample_aligned_data):
        """Test team-focused momentum detection."""
        features_df, events = detector.detect_momentum(sample_aligned_data, team_focus='DAL')
        
        # Should still work with team focus
        assert len(features_df) == len(sample_aligned_data)
        assert isinstance(events, list)

    def test_edge_cases(self, detector):
        """Test edge cases and error conditions."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'yards_gained': [5],
            'close_price': [50.0],
            'volume': [1000],
            'score_differential': [0],
            'field_position': [50],
            'quarter': [1],
            'time_remaining': [3600]
        })
        
        features_df, events = detector.detect_momentum(minimal_data)
        assert len(features_df) == 1
        
        # Test with missing optional columns
        sparse_data = pd.DataFrame({
            'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(5)],
            'play_type': ['pass'] * 5,
            'yards_gained': [5] * 5,
            'close_price': [50.0] * 5,
            'volume': [1000] * 5
        })
        
        features_df, events = detector.detect_momentum(sparse_data)
        assert len(features_df) == 5

    def test_feature_importance_mapping(self, detector):
        """Test feature importance mapping."""
        mapping = detector.get_feature_importance_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check some expected mappings
        expected_features = [
            'game_momentum_score_trend',
            'price_momentum_strength',
            'volume_momentum',
            'combined_momentum',
            'momentum_state_strength'
        ]
        
        for feature in expected_features:
            assert feature in mapping
            assert isinstance(mapping[feature], str)
            assert len(mapping[feature]) > 0

    def test_momentum_event_properties(self, detector):
        """Test MomentumEvent object properties."""
        timestamp = datetime.now()
        event = MomentumEvent(
            timestamp=timestamp,
            event_type=EventType.TOUCHDOWN,
            momentum_type=MomentumType.GAME_MOMENTUM,
            direction=MomentumDirection.BULLISH,
            strength=0.8,
            confidence=0.9,
            duration_estimate=timedelta(minutes=5),
            description="Test touchdown",
            game_context={'quarter': 4},
            price_context={'price_change': 0.02}
        )
        
        assert event.timestamp == timestamp
        assert event.event_type == EventType.TOUCHDOWN
        assert event.momentum_type == MomentumType.GAME_MOMENTUM
        assert event.direction == MomentumDirection.BULLISH
        assert event.strength == 0.8
        assert event.confidence == 0.9
        assert event.description == "Test touchdown"
        assert event.game_context['quarter'] == 4
        assert event.price_context['price_change'] == 0.02

    def test_performance_with_large_dataset(self, detector):
        """Test performance with larger dataset."""
        # Create larger dataset
        base_time = datetime.now()
        n_points = 500
        
        large_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i) for i in range(n_points)],
            'play_type': ['pass', 'run', 'touchdown', 'punt'] * (n_points // 4),
            'yards_gained': [5, 8, 25, -2] * (n_points // 4),
            'close_price': 50 + np.cumsum(np.random.randn(n_points) * 0.01),
            'volume': np.random.randint(1000, 5000, n_points),
            'score_differential': [0] * n_points,
            'field_position': [50] * n_points,
            'quarter': [1] * n_points,
            'time_remaining': [3600] * n_points
        })
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        features_df, events = detector.detect_momentum(large_data)
        end_time = time.time()
        
        assert end_time - start_time < 20  # Should complete within 20 seconds
        assert len(features_df) == n_points