"""Unit tests for NFLDataProcessor."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from src.nfl_trading.data.nfl_processor import NFLDataProcessor, PlayEvent, PlayEventValidator


class TestNFLDataProcessor:
    """Test cases for NFLDataProcessor."""

    def test_initialization(self, test_config):
        """Test processor initialization."""
        processor = NFLDataProcessor(test_config)
        assert processor.config == test_config
        assert processor.logger is not None
        assert isinstance(processor.momentum_features, list)

    def test_load_json_file_success(self, sample_nfl_data, temp_json_file):
        """Test successful JSON file loading."""
        processor = NFLDataProcessor()
        json_file = temp_json_file(sample_nfl_data)

        loaded_data = processor.load_json_file(json_file)

        assert loaded_data == sample_nfl_data
        assert 'game' in loaded_data
        assert 'plays' in loaded_data

    def test_load_json_file_not_found(self):
        """Test JSON file loading with non-existent file."""
        processor = NFLDataProcessor()

        with pytest.raises(FileNotFoundError):
            processor.load_json_file("non_existent_file.json")

    def test_load_json_file_invalid_json(self, tmp_path):
        """Test JSON file loading with invalid JSON."""
        processor = NFLDataProcessor()

        # Create invalid JSON file
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            processor.load_json_file(invalid_json_file)

    def test_parse_play_events_success(self, sample_nfl_data):
        """Test successful play event parsing."""
        processor = NFLDataProcessor()

        plays = processor.parse_play_events(sample_nfl_data)

        assert isinstance(plays, list)
        assert len(plays) == len(sample_nfl_data['plays'])

        for play in plays:
            assert isinstance(play, PlayEvent)
            assert play.game_id == sample_nfl_data['game']['id']
            assert play.home_team == sample_nfl_data['game']['home_team']
            assert play.away_team == sample_nfl_data['game']['away_team']

    def test_parse_play_events_empty_plays(self):
        """Test parsing with empty plays list."""
        processor = NFLDataProcessor()

        empty_data = {
            'game': {'id': 'test_game', 'home_team': 'HOME', 'away_team': 'AWAY'},
            'plays': []
        }

        plays = processor.parse_play_events(empty_data)
        assert plays == []

    def test_parse_single_play_valid(self):
        """Test parsing a single valid play."""
        processor = NFLDataProcessor()

        play_json = {
            'id': 'play_1',
            'timestamp': '2024-01-15T13:00:00Z',
            'type': 'pass',
            'down': 1,
            'distance': 10,
            'field_position': 50,
            'score_home': 7,
            'score_away': 0,
            'time_remaining': 3600,
            'quarter': 1,
            'possession_team': 'HOME',
            'description': 'Test play',
            'result': 'success',
            'yards_gained': 15
        }

        play = processor._parse_single_play(play_json, 'game_1', 'HOME', 'AWAY')

        assert play is not None
        assert isinstance(play, PlayEvent)
        assert play.play_id == 'play_1'
        assert play.play_type == 'pass'
        assert play.down == 1
        assert play.yards_gained == 15

    def test_parse_single_play_invalid(self):
        """Test parsing an invalid play."""
        processor = NFLDataProcessor()

        # Missing required fields
        invalid_play = {
            'id': 'play_1',
            'type': 'pass'
            # Missing other required fields
        }

        play = processor._parse_single_play(invalid_play, 'game_1', 'HOME', 'AWAY')
        assert play is None

    def test_calculate_momentum_features(self):
        """Test momentum feature calculation."""
        processor = NFLDataProcessor()

        # Create sample plays
        base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
        plays = []

        for i in range(5):
            play = PlayEvent(
                timestamp=base_time + timedelta(minutes=i*2),
                game_id='game_1',
                play_id=f'play_{i}',
                play_type='pass',
                down=1,
                distance=10,
                field_position=50 + (i * 5),
                score_home=i * 3,
                score_away=0,
                time_remaining=3600 - (i * 120),
                quarter=1,
                possession_team='HOME',
                home_team='HOME',
                away_team='AWAY',
                description=f'Play {i}',
                yards_gained=5 + i
            )
            plays.append(play)

        plays_with_momentum = processor.calculate_momentum_features(plays)

        for play in plays_with_momentum:
            assert play.momentum_score is not None
            assert -1.0 <= play.momentum_score <= 1.0

    def test_calculate_momentum_score(self):
        """Test momentum score calculation for specific scenarios."""
        processor = NFLDataProcessor()

        base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

        # Create plays with improving field position and successful outcomes
        plays = []
        for i in range(3):
            play = PlayEvent(
                timestamp=base_time + timedelta(minutes=i*2),
                game_id='game_1',
                play_id=f'play_{i}',
                play_type='pass',
                score_home=i * 7,  # Increasing score
                score_away=0,
                field_position=30 + (i * 10),  # Improving field position
                time_remaining=3600 - (i * 120),
                quarter=1,
                possession_team='HOME',
                home_team='HOME',
                away_team='AWAY',
                description=f'Play {i}',
                yards_gained=10  # Successful plays
            )
            plays.append(play)

        # Calculate momentum for last play
        momentum = processor._calculate_momentum_score(plays, 2)

        # Should be positive due to score advantage and successful recent plays
        assert momentum > 0

    def test_to_dataframe(self, sample_nfl_data):
        """Test conversion to DataFrame."""
        processor = NFLDataProcessor()

        plays = processor.parse_play_events(sample_nfl_data)
        plays = processor.calculate_momentum_features(plays)

        df = processor.to_dataframe(plays)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(plays)
        assert 'timestamp' in df.columns
        assert 'momentum_score' in df.columns
        assert 'score_differential' in df.columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

        # Check sorted by timestamp
        assert df['timestamp'].is_monotonic_increasing

    def test_to_dataframe_empty(self):
        """Test DataFrame conversion with empty plays."""
        processor = NFLDataProcessor()

        df = processor.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_save_processed_data_parquet(self, sample_nfl_dataframe, tmp_path):
        """Test saving data to Parquet format."""
        processor = NFLDataProcessor()
        output_path = tmp_path / "test_output.parquet"

        processor.save_processed_data(sample_nfl_dataframe, output_path)

        assert output_path.exists()

        # Verify data can be loaded back
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded_df, sample_nfl_dataframe)

    def test_save_processed_data_csv(self, sample_nfl_dataframe, tmp_path):
        """Test saving data to CSV format."""
        processor = NFLDataProcessor()
        output_path = tmp_path / "test_output.csv"

        processor.save_processed_data(sample_nfl_dataframe, output_path)

        assert output_path.exists()

    def test_save_processed_data_json(self, sample_nfl_dataframe, tmp_path):
        """Test saving data to JSON format."""
        processor = NFLDataProcessor()
        output_path = tmp_path / "test_output.json"

        processor.save_processed_data(sample_nfl_dataframe, output_path)

        assert output_path.exists()

    def test_save_processed_data_unsupported_format(self, sample_nfl_dataframe, tmp_path):
        """Test saving data with unsupported format."""
        processor = NFLDataProcessor()
        output_path = tmp_path / "test_output.xyz"

        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.save_processed_data(sample_nfl_dataframe, output_path)

    def test_process_file_end_to_end(self, sample_nfl_data, temp_json_file, tmp_path):
        """Test complete file processing workflow."""
        processor = NFLDataProcessor()

        input_file = temp_json_file(sample_nfl_data)
        output_file = tmp_path / "processed_output.parquet"

        df = processor.process_file(input_file, output_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert output_file.exists()

        # Check required columns
        required_columns = ['timestamp', 'game_id', 'play_id', 'momentum_score']
        for col in required_columns:
            assert col in df.columns

    def test_process_directory(self, sample_nfl_data, temp_directory, tmp_path):
        """Test processing multiple files in directory."""
        processor = NFLDataProcessor()

        input_dir = temp_directory(['input'])
        output_dir = tmp_path / "output"

        # Create multiple JSON files
        for i in range(3):
            test_data = sample_nfl_data.copy()
            test_data['game']['id'] = f'game_{i}'

            json_file = input_dir / "input" / f"game_{i}.json"
            with open(json_file, 'w') as f:
                json.dump(test_data, f)

        dataframes = processor.process_directory(input_dir / "input", output_dir)

        assert len(dataframes) == 3
        assert all(isinstance(df, pd.DataFrame) for df in dataframes)
        assert len(list(output_dir.glob("*.parquet"))) == 3

    def test_process_directory_empty(self, temp_directory, tmp_path):
        """Test processing empty directory."""
        processor = NFLDataProcessor()

        input_dir = temp_directory()
        output_dir = tmp_path / "output"

        dataframes = processor.process_directory(input_dir, output_dir)

        assert dataframes == []


class TestPlayEventValidator:
    """Test cases for PlayEventValidator."""

    def test_valid_play_event(self):
        """Test validation of valid play event."""
        valid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'game_id': 'game_1',
            'play_id': 'play_1',
            'play_type': 'pass',
            'score_home': 7,
            'score_away': 0,
            'time_remaining': 3600,
            'quarter': 1,
            'possession_team': 'HOME',
            'home_team': 'HOME',
            'away_team': 'AWAY',
            'description': 'Test play'
        }

        validator = PlayEventValidator(**valid_data)
        assert validator.game_id == 'game_1'
        assert validator.quarter == 1

    def test_invalid_quarter(self):
        """Test validation with invalid quarter."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'game_id': 'game_1',
            'play_id': 'play_1',
            'play_type': 'pass',
            'score_home': 7,
            'score_away': 0,
            'time_remaining': 3600,
            'quarter': 6,  # Invalid
            'possession_team': 'HOME',
            'home_team': 'HOME',
            'away_team': 'AWAY',
            'description': 'Test play'
        }

        with pytest.raises(ValueError, match="Invalid quarter"):
            PlayEventValidator(**invalid_data)

    def test_invalid_down(self):
        """Test validation with invalid down."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'game_id': 'game_1',
            'play_id': 'play_1',
            'play_type': 'pass',
            'down': 5,  # Invalid
            'score_home': 7,
            'score_away': 0,
            'time_remaining': 3600,
            'quarter': 1,
            'possession_team': 'HOME',
            'home_team': 'HOME',
            'away_team': 'AWAY',
            'description': 'Test play'
        }

        with pytest.raises(ValueError, match="Invalid down"):
            PlayEventValidator(**invalid_data)

    def test_invalid_field_position(self):
        """Test validation with invalid field position."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'game_id': 'game_1',
            'play_id': 'play_1',
            'play_type': 'pass',
            'field_position': 150,  # Invalid
            'score_home': 7,
            'score_away': 0,
            'time_remaining': 3600,
            'quarter': 1,
            'possession_team': 'HOME',
            'home_team': 'HOME',
            'away_team': 'AWAY',
            'description': 'Test play'
        }

        with pytest.raises(ValueError, match="Invalid field position"):
            PlayEventValidator(**invalid_data)