"""Unit tests for DataAligner."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json

from src.nfl_trading.data.data_aligner import DataAligner, AlignmentMethod, AlignmentResult


class TestDataAligner:
    """Test cases for DataAligner."""

    def test_initialization(self, test_config):
        """Test aligner initialization."""
        aligner = DataAligner(test_config)
        assert aligner.config == test_config
        assert aligner.logger is not None
        assert isinstance(aligner.time_tolerance, timedelta)
        assert aligner.default_alignment_method == AlignmentMethod.NEAREST

    def test_validate_nfl_data_success(self, sample_nfl_dataframe):
        """Test successful NFL data validation."""
        aligner = DataAligner()

        validated_df = aligner._validate_nfl_data(sample_nfl_dataframe)

        assert isinstance(validated_df, pd.DataFrame)
        assert 'timestamp' in validated_df.columns
        assert pd.api.types.is_datetime64_any_dtype(validated_df['timestamp'])
        assert validated_df['timestamp'].is_monotonic_increasing

    def test_validate_nfl_data_missing_timestamp(self):
        """Test NFL data validation with missing timestamp column."""
        aligner = DataAligner()

        invalid_df = pd.DataFrame({'play_id': ['play_1', 'play_2']})

        with pytest.raises(ValueError, match="NFL data missing required columns"):
            aligner._validate_nfl_data(invalid_df)

    def test_validate_price_data_success(self, sample_kalshi_dataframe):
        """Test successful price data validation."""
        aligner = DataAligner()

        validated_df = aligner._validate_price_data(sample_kalshi_dataframe)

        assert isinstance(validated_df, pd.DataFrame)
        assert 'timestamp' in validated_df.columns
        assert pd.api.types.is_datetime64_any_dtype(validated_df['timestamp'])
        assert validated_df['timestamp'].is_monotonic_increasing

    def test_validate_price_data_missing_timestamp(self):
        """Test price data validation with missing timestamp column."""
        aligner = DataAligner()

        invalid_df = pd.DataFrame({'price': [0.5, 0.6]})

        with pytest.raises(ValueError, match="Price data missing required columns"):
            aligner._validate_price_data(invalid_df)

    def test_get_default_price_columns(self, sample_kalshi_dataframe):
        """Test getting default price columns."""
        aligner = DataAligner()

        price_columns = aligner._get_default_price_columns(sample_kalshi_dataframe)

        assert isinstance(price_columns, list)
        assert 'close_price' in price_columns
        assert len(price_columns) > 0

        # Should only include columns that exist in the dataframe
        for col in price_columns:
            assert col in sample_kalshi_dataframe.columns

    def test_align_data_nearest_method(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test data alignment using nearest method."""
        aligner = DataAligner()

        result = aligner.align_data(
            sample_nfl_dataframe,
            sample_kalshi_dataframe,
            method=AlignmentMethod.NEAREST,
            time_tolerance=timedelta(minutes=5)
        )

        assert isinstance(result, AlignmentResult)
        assert isinstance(result.aligned_data, pd.DataFrame)
        assert isinstance(result.alignment_stats, dict)
        assert len(result.aligned_data) > 0

        # Check that price columns are included
        price_columns = [col for col in result.aligned_data.columns if col.startswith('price_')]
        assert len(price_columns) > 0

    def test_align_data_forward_fill_method(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test data alignment using forward fill method."""
        aligner = DataAligner()

        result = aligner.align_data(
            sample_nfl_dataframe,
            sample_kalshi_dataframe,
            method=AlignmentMethod.FORWARD_FILL,
            time_tolerance=timedelta(minutes=5)
        )

        assert isinstance(result, AlignmentResult)
        assert isinstance(result.aligned_data, pd.DataFrame)

    def test_align_data_backward_fill_method(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test data alignment using backward fill method."""
        aligner = DataAligner()

        result = aligner.align_data(
            sample_nfl_dataframe,
            sample_kalshi_dataframe,
            method=AlignmentMethod.BACKWARD_FILL,
            time_tolerance=timedelta(minutes=5)
        )

        assert isinstance(result, AlignmentResult)
        assert isinstance(result.aligned_data, pd.DataFrame)

    def test_align_data_interpolation_method(self):
        """Test data alignment using interpolation method."""
        aligner = DataAligner()

        # Create test data with precise timestamps for interpolation
        base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

        # NFL data - plays at specific times
        nfl_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=2.5), base_time + timedelta(minutes=7.5)],
            'play_id': ['play_1', 'play_2'],
            'momentum_score': [0.5, -0.3]
        })

        # Kalshi data - prices at different intervals
        kalshi_data = pd.DataFrame({
            'timestamp': [base_time, base_time + timedelta(minutes=5), base_time + timedelta(minutes=10)],
            'close_price': [0.4, 0.6, 0.5],
            'volume': [1000, 1500, 1200]
        })

        result = aligner.align_data(
            nfl_data,
            kalshi_data,
            method=AlignmentMethod.INTERPOLATION,
            time_tolerance=timedelta(minutes=5)
        )

        assert isinstance(result, AlignmentResult)
        assert len(result.aligned_data) > 0

        # Interpolated values should be between the surrounding data points
        if len(result.aligned_data) > 0:
            interpolated_price = result.aligned_data.iloc[0]['close_price']
            assert 0.4 <= interpolated_price <= 0.6

    def test_align_data_no_matches(self):
        """Test data alignment when no matches are found."""
        aligner = DataAligner()

        # Create data with non-overlapping time ranges
        base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

        nfl_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=1)],
            'play_id': ['play_1']
        })

        kalshi_data = pd.DataFrame({
            'timestamp': [base_time],
            'close_price': [0.5]
        })

        result = aligner.align_data(
            nfl_data,
            kalshi_data,
            time_tolerance=timedelta(minutes=10)
        )

        # Should have no aligned data
        assert len(result.aligned_data) == 0
        assert len(result.unmatched_plays) == 1

    def test_create_aligned_record(self):
        """Test creation of aligned record."""
        aligner = DataAligner()

        # Create sample play and price data
        play = pd.Series({
            'timestamp': datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
            'play_id': 'play_1',
            'momentum_score': 0.5
        })

        price = pd.Series({
            'timestamp': datetime(2024, 1, 15, 13, 1, 0, tzinfo=timezone.utc),
            'close_price': 0.6,
            'volume': 1000
        })

        time_diff = timedelta(minutes=1)
        price_columns = ['close_price', 'volume']

        record = aligner._create_aligned_record(play, price, price_columns, time_diff)

        assert isinstance(record, dict)
        assert record['play_id'] == 'play_1'
        assert record['momentum_score'] == 0.5
        assert record['price_close_price'] == 0.6
        assert record['price_volume'] == 1000
        assert record['time_diff'] == 60.0  # seconds

    def test_interpolate_price_data(self):
        """Test price data interpolation."""
        aligner = DataAligner()

        # Create before and after price data
        base_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

        before_price = pd.Series({
            'timestamp': base_time,
            'close_price': 0.4,
            'volume': 1000
        })

        after_price = pd.Series({
            'timestamp': base_time + timedelta(minutes=10),
            'close_price': 0.6,
            'volume': 1500
        })

        target_time = base_time + timedelta(minutes=5)  # Halfway point
        numeric_columns = ['close_price', 'volume']

        interpolated = aligner._interpolate_price_data(
            before_price, after_price, target_time, numeric_columns
        )

        # Should interpolate to halfway points
        assert abs(interpolated['close_price'] - 0.5) < 0.01
        assert abs(interpolated['volume'] - 1250) < 1

    def test_calculate_alignment_stats(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test alignment statistics calculation."""
        aligner = DataAligner()

        # Create sample aligned data
        aligned_data = pd.DataFrame({
            'timestamp': sample_nfl_dataframe['timestamp'][:5],
            'play_id': sample_nfl_dataframe['play_id'][:5],
            'time_diff': [30, 45, 60, 20, 40]  # seconds
        })

        stats = aligner._calculate_alignment_stats(
            aligned_data, sample_nfl_dataframe, sample_kalshi_dataframe
        )

        assert isinstance(stats, dict)
        assert 'total_nfl_plays' in stats
        assert 'total_price_points' in stats
        assert 'aligned_records' in stats
        assert 'alignment_rate' in stats
        assert 'time_diff_stats' in stats

        assert stats['total_nfl_plays'] == len(sample_nfl_dataframe)
        assert stats['total_price_points'] == len(sample_kalshi_dataframe)
        assert stats['aligned_records'] == len(aligned_data)
        assert 0 <= stats['alignment_rate'] <= 1

    def test_create_feature_matrix(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test feature matrix creation."""
        aligner = DataAligner()

        # First align the data
        result = aligner.align_data(
            sample_nfl_dataframe,
            sample_kalshi_dataframe,
            time_tolerance=timedelta(minutes=10)
        )

        if len(result.aligned_data) > 5:  # Need enough data for lookback window
            feature_matrix = aligner.create_feature_matrix(
                result.aligned_data,
                lookback_window=3
            )

            assert isinstance(feature_matrix, pd.DataFrame)
            assert len(feature_matrix) == len(result.aligned_data) - 3  # Due to lookback window

            # Check for current and lagged features
            current_features = [col for col in feature_matrix.columns if col.startswith('current_')]
            lag_features = [col for col in feature_matrix.columns if col.startswith('lag_')]

            assert len(current_features) > 0
            assert len(lag_features) > 0

    def test_save_aligned_data(self, sample_nfl_dataframe, sample_kalshi_dataframe, tmp_path):
        """Test saving aligned data."""
        aligner = DataAligner()

        result = aligner.align_data(
            sample_nfl_dataframe,
            sample_kalshi_dataframe,
            time_tolerance=timedelta(minutes=10)
        )

        output_path = tmp_path / "test_alignment.parquet"

        aligner.save_aligned_data(result, output_path)

        # Check that files were created
        aligned_file = tmp_path / "test_alignment_aligned.parquet"
        stats_file = tmp_path / "test_alignment_alignment_stats.json"

        assert aligned_file.exists()
        assert stats_file.exists()

        # Check that data can be loaded back
        loaded_df = pd.read_parquet(aligned_file)
        assert len(loaded_df) == len(result.aligned_data)

        # Check stats file
        with open(stats_file, 'r') as f:
            loaded_stats = json.load(f)
        assert isinstance(loaded_stats, dict)

    def test_make_json_serializable(self):
        """Test JSON serialization helper."""
        aligner = DataAligner()

        test_data = {
            'timestamp': datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
            'timedelta': timedelta(minutes=5),
            'integer': np.int64(42),
            'float': np.float64(3.14),
            'nested': {
                'timestamp': pd.Timestamp('2024-01-15 13:00:00+00:00')
            }
        }

        serializable = aligner._make_json_serializable(test_data)

        # Should be JSON serializable
        import json
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)

    def test_unknown_alignment_method(self, sample_nfl_dataframe, sample_kalshi_dataframe):
        """Test alignment with unknown method."""
        aligner = DataAligner()

        with pytest.raises(ValueError, match="Unknown alignment method"):
            aligner.align_data(
                sample_nfl_dataframe,
                sample_kalshi_dataframe,
                method="invalid_method"  # Invalid method
            )


class TestAlignmentResult:
    """Test cases for AlignmentResult."""

    def test_alignment_result_creation(self):
        """Test creation of AlignmentResult object."""
        aligned_data = pd.DataFrame({'test': [1, 2, 3]})
        stats = {'test_stat': 42}
        unmatched_plays = pd.DataFrame({'play': ['a', 'b']})
        unmatched_prices = pd.DataFrame({'price': [0.5, 0.6]})

        result = AlignmentResult(
            aligned_data=aligned_data,
            alignment_stats=stats,
            unmatched_plays=unmatched_plays,
            unmatched_prices=unmatched_prices
        )

        assert result.aligned_data.equals(aligned_data)
        assert result.alignment_stats == stats
        assert result.unmatched_plays.equals(unmatched_plays)
        assert result.unmatched_prices.equals(unmatched_prices)