"""Tests for GameStateExtractor class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.nfl_trading.features.game_state_extractor import GameStateExtractor, GameSituation, FieldZone


class TestGameStateExtractor:
    """Test cases for GameStateExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create GameStateExtractor instance."""
        return GameStateExtractor()

    @pytest.fixture
    def sample_plays_data(self):
        """Create sample NFL plays data."""
        base_time = datetime.now()
        
        data = {
            'timestamp': [base_time + timedelta(minutes=i) for i in range(10)],
            'play_type': ['pass', 'run', 'touchdown', 'punt', 'field_goal', 
                         'interception', 'sack', 'penalty', 'timeout', 'run'],
            'possession_team': ['DAL', 'DAL', 'DAL', 'PHI', 'PHI', 
                               'DAL', 'PHI', 'DAL', 'DAL', 'PHI'],
            'field_position': [25, 35, 85, 45, 75, 60, 20, 50, 90, 15],
            'score_home': [0, 0, 7, 7, 7, 7, 10, 10, 10, 10],
            'score_away': [0, 0, 0, 0, 3, 3, 3, 3, 3, 10],
            'quarter': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            'time_remaining': [3600, 3540, 3480, 2700, 2640, 2580, 1800, 1740, 300, 240],
            'down': [1, 2, 1, 4, 1, 3, 2, 1, 1, 2],
            'distance': [10, 7, 10, 8, 10, 5, 12, 10, 10, 6],
            'yards_gained': [8, 12, 25, -2, 0, -5, 8, 0, 15, 7],
            'home_team': ['DAL'] * 10,
            'away_team': ['PHI'] * 10
        }
        
        return pd.DataFrame(data)

    def test_initialization(self, extractor):
        """Test GameStateExtractor initialization."""
        assert extractor.lookback_window == 5
        assert extractor.drive_lookback == 3
        assert extractor.red_zone_threshold == 20
        assert extractor.long_play_threshold == 20
        assert extractor.short_yardage_threshold == 3

    def test_extract_features_basic(self, extractor, sample_plays_data):
        """Test basic feature extraction."""
        features_df = extractor.extract_features(sample_plays_data)
        
        # Check that features were extracted
        assert len(features_df) == len(sample_plays_data)
        assert 'score_differential' in features_df.columns
        assert 'distance_to_endzone' in features_df.columns
        assert 'game_situation' in features_df.columns
        assert 'field_zone' in features_df.columns

    def test_validate_plays_data(self, extractor):
        """Test plays data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'possession_team': ['DAL'],
            'field_position': [50],
            'score_home': [7],
            'score_away': [0],
            'quarter': [1],
            'time_remaining': [3600]
        })
        
        validated = extractor._validate_plays_data(valid_data)
        assert len(validated) == 1
        assert 'down' in validated.columns
        assert 'distance' in validated.columns
        
        # Missing required columns
        invalid_data = pd.DataFrame({'timestamp': [datetime.now()]})
        with pytest.raises(ValueError, match="Missing required columns"):
            extractor._validate_plays_data(invalid_data)

    def test_extract_base_features(self, extractor, sample_plays_data):
        """Test base feature extraction."""
        validated_data = extractor._validate_plays_data(sample_plays_data)
        features_df = extractor._extract_base_features(validated_data)
        
        # Check base features
        assert 'score_differential' in features_df.columns
        assert 'absolute_score_diff' in features_df.columns
        assert 'distance_to_endzone' in features_df.columns
        assert 'game_time_elapsed' in features_df.columns
        
        # Check first row values
        first_row = features_df.iloc[0]
        assert first_row['score_differential'] == 0  # 0 - 0
        assert first_row['distance_to_endzone'] == 75  # 100 - 25
        assert first_row['field_position'] == 25

    def test_categorical_features(self, extractor, sample_plays_data):
        """Test categorical feature generation."""
        play = sample_plays_data.iloc[2]  # Touchdown play
        categorical_features = extractor._get_categorical_features(play)
        
        # Check game situation (DAL 7 - PHI 0 = +7)
        assert categorical_features['game_situation'] == GameSituation.SLIGHT_AHEAD.value
        
        # Check field zone (field_position = 85)
        assert categorical_features['field_zone'] == FieldZone.RED_ZONE.value
        
        # Check red zone flag
        assert categorical_features['in_red_zone'] is True
        
        # Check down situation
        assert categorical_features['down_situation'] == 'first_down'

    def test_game_situation_classification(self, extractor):
        """Test game situation classification logic."""
        # Test different score differentials
        test_cases = [
            (20, GameSituation.BLOWOUT_AHEAD.value),
            (10, GameSituation.COMFORTABLE_AHEAD.value), 
            (3, GameSituation.SLIGHT_AHEAD.value),
            (0, GameSituation.TIED.value),
            (-3, GameSituation.SLIGHT_BEHIND.value),
            (-10, GameSituation.COMFORTABLE_BEHIND.value),
            (-20, GameSituation.BLOWOUT_BEHIND.value)
        ]
        
        for score_diff, expected_situation in test_cases:
            play = pd.Series({
                'score_home': 14 + score_diff,
                'score_away': 14,
                'field_position': 50,
                'down': 1,
                'distance': 10,
                'quarter': 2,
                'time_remaining': 1800
            })
            
            features = extractor._get_categorical_features(play)
            assert features['game_situation'] == expected_situation

    def test_field_zone_classification(self, extractor):
        """Test field zone classification logic."""
        test_cases = [
            (5, FieldZone.OWN_ENDZONE.value),
            (25, FieldZone.OWN_TERRITORY.value),
            (50, FieldZone.MIDFIELD.value),
            (65, FieldZone.OPPONENT_TERRITORY.value),
            (85, FieldZone.RED_ZONE.value)
        ]
        
        for field_pos, expected_zone in test_cases:
            play = pd.Series({
                'score_home': 7,
                'score_away': 7,
                'field_position': field_pos,
                'down': 1,
                'distance': 10,
                'quarter': 2,
                'time_remaining': 1800
            })
            
            features = extractor._get_categorical_features(play)
            assert features['field_zone'] == expected_zone

    def test_drive_identification(self, extractor, sample_plays_data):
        """Test drive identification logic."""
        drives = extractor._identify_drives(sample_plays_data)
        
        # Should have multiple drives due to possession changes
        assert len(drives) > 1
        
        # Check first drive
        first_drive = drives[0]
        assert first_drive.starting_field_position == 25
        assert len(first_drive.plays) >= 1

    def test_momentum_features(self, extractor, sample_plays_data):
        """Test momentum feature calculation."""
        validated_data = extractor._validate_plays_data(sample_plays_data)
        features_df = extractor._extract_base_features(validated_data)
        features_df = extractor._add_momentum_features(features_df, validated_data)
        
        # Check momentum features exist
        momentum_cols = [col for col in features_df.columns if 'momentum_' in col]
        assert len(momentum_cols) > 0
        
        expected_momentum_features = [
            'momentum_yards_trend',
            'momentum_success_rate', 
            'momentum_big_play_rate',
            'momentum_consecutive_success'
        ]
        
        for feature in expected_momentum_features:
            assert feature in features_df.columns

    def test_situational_features(self, extractor, sample_plays_data):
        """Test situational feature calculation."""
        validated_data = extractor._validate_plays_data(sample_plays_data)
        features_df = extractor._extract_base_features(validated_data)
        features_df = extractor._add_situational_features(features_df, validated_data)
        
        # Check situational features exist
        situational_cols = [col for col in features_df.columns if 'situation_' in col]
        assert len(situational_cols) > 0
        
        expected_situational_features = [
            'situation_goal_line_situation',
            'situation_red_zone_situation',
            'situation_short_yardage',
            'situation_passing_situation',
            'situation_two_minute_drill'
        ]
        
        for feature in expected_situational_features:
            assert feature in features_df.columns

    def test_time_features(self, extractor, sample_plays_data):
        """Test time-based feature calculation."""
        validated_data = extractor._validate_plays_data(sample_plays_data)
        features_df = extractor._extract_base_features(validated_data)
        features_df = extractor._add_time_features(features_df, validated_data)
        
        # Check time features exist
        time_cols = [col for col in features_df.columns if 'time_' in col]
        assert len(time_cols) > 0
        
        expected_time_features = [
            'time_game_progress',
            'time_remaining_pct',
            'time_first_half',
            'time_final_quarter',
            'time_hurry_up_situation'
        ]
        
        for feature in expected_time_features:
            assert feature in features_df.columns

    def test_team_specific_features(self, extractor, sample_plays_data):
        """Test team-specific feature calculation."""
        validated_data = extractor._validate_plays_data(sample_plays_data)
        features_df = extractor._extract_base_features(validated_data)
        features_df = extractor._add_team_specific_features(features_df, validated_data, 'DAL')
        
        # Check team features exist
        team_cols = [col for col in features_df.columns if 'team_' in col]
        assert len(team_cols) > 0
        
        expected_team_features = [
            'team_has_possession',
            'team_score',
            'team_score_differential',
            'team_winning',
            'team_field_position'
        ]
        
        for feature in expected_team_features:
            assert feature in features_df.columns

    def test_empty_data_handling(self, extractor):
        """Test handling of empty or minimal data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            extractor.extract_features(empty_df)
        
        # Single row data
        single_row_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'possession_team': ['DAL'],
            'field_position': [50],
            'score_home': [7],
            'score_away': [0],
            'quarter': [1],
            'time_remaining': [3600]
        })
        
        features_df = extractor.extract_features(single_row_data)
        assert len(features_df) == 1

    def test_feature_importance_mapping(self, extractor):
        """Test feature importance mapping."""
        mapping = extractor.get_feature_importance_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check some expected mappings
        expected_features = [
            'score_differential',
            'field_position', 
            'distance_to_endzone',
            'game_situation',
            'momentum_yards_trend'
        ]
        
        for feature in expected_features:
            assert feature in mapping
            assert isinstance(mapping[feature], str)
            assert len(mapping[feature]) > 0

    def test_edge_cases(self, extractor):
        """Test edge cases and error conditions."""
        # Test with NaN values
        data_with_nans = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(minutes=1)],
            'play_type': ['pass', 'run'],
            'possession_team': ['DAL', 'PHI'],
            'field_position': [50, np.nan],
            'score_home': [7, 7],
            'score_away': [0, np.nan],
            'quarter': [1, 1],
            'time_remaining': [3600, 3540]
        })
        
        features_df = extractor.extract_features(data_with_nans)
        assert len(features_df) == 2
        
        # Should handle NaN values gracefully
        assert not features_df.isnull().all().any()

    def test_performance_with_large_dataset(self, extractor):
        """Test performance with larger dataset."""
        # Create larger dataset
        base_time = datetime.now()
        large_data = {
            'timestamp': [base_time + timedelta(minutes=i) for i in range(100)],
            'play_type': ['pass', 'run'] * 50,
            'possession_team': ['DAL', 'PHI'] * 50,
            'field_position': list(range(1, 101)),
            'score_home': [i // 10 for i in range(100)],
            'score_away': [i // 15 for i in range(100)],
            'quarter': [1 + (i // 25) for i in range(100)],
            'time_remaining': [3600 - i * 30 for i in range(100)],
            'down': [1 + (i % 4) for i in range(100)],
            'distance': [10 - (i % 11) for i in range(100)],
            'yards_gained': [(i % 21) - 5 for i in range(100)],
            'home_team': ['DAL'] * 100,
            'away_team': ['PHI'] * 100
        }
        
        large_df = pd.DataFrame(large_data)
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        features_df = extractor.extract_features(large_df)
        end_time = time.time()
        
        assert end_time - start_time < 10  # Should complete within 10 seconds
        assert len(features_df) == 100