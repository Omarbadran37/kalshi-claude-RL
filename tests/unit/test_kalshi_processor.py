"""Unit tests for KalshiDataProcessor."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from src.nfl_trading.data.kalshi_processor import KalshiDataProcessor, Candlestick, CandlestickValidator


class TestKalshiDataProcessor:
    """Test cases for KalshiDataProcessor."""

    def test_initialization(self, test_config):
        """Test processor initialization."""
        processor = KalshiDataProcessor(test_config)
        assert processor.config == test_config
        assert processor.logger is not None
        assert isinstance(processor.technical_indicators, dict)

    def test_load_json_file_success(self, sample_kalshi_data, temp_json_file):
        """Test successful JSON file loading."""
        processor = KalshiDataProcessor()
        json_file = temp_json_file(sample_kalshi_data)

        loaded_data = processor.load_json_file(json_file)

        assert loaded_data == sample_kalshi_data
        assert 'market_id' in loaded_data
        assert 'candlesticks' in loaded_data

    def test_load_json_file_not_found(self):
        """Test JSON file loading with non-existent file."""
        processor = KalshiDataProcessor()

        with pytest.raises(FileNotFoundError):
            processor.load_json_file("non_existent_file.json")

    def test_parse_candlesticks_success(self, sample_kalshi_data):
        """Test successful candlestick parsing."""
        processor = KalshiDataProcessor()

        candlesticks = processor.parse_candlesticks(sample_kalshi_data)

        assert isinstance(candlesticks, list)
        assert len(candlesticks) == len(sample_kalshi_data['candlesticks'])

        for candlestick in candlesticks:
            assert isinstance(candlestick, Candlestick)
            assert candlestick.market_id == sample_kalshi_data['market_id']
            assert 0 <= candlestick.open_price <= 1
            assert 0 <= candlestick.close_price <= 1

    def test_parse_candlesticks_empty(self):
        """Test parsing with empty candlesticks list."""
        processor = KalshiDataProcessor()

        empty_data = {
            'market_id': 'test_market',
            'candlesticks': []
        }

        candlesticks = processor.parse_candlesticks(empty_data)
        assert candlesticks == []

    def test_parse_single_candlestick_valid(self):
        """Test parsing a single valid candlestick."""
        processor = KalshiDataProcessor()

        candlestick_json = {
            'timestamp': '2024-01-15T13:00:00Z',
            'open': 0.5,
            'high': 0.55,
            'low': 0.45,
            'close': 0.52,
            'volume': 1000,
            'bid': 0.51,
            'ask': 0.53,
            'bid_size': 50,
            'ask_size': 60,
            'trades': 15,
            'vwap': 0.515
        }

        candlestick = processor._parse_single_candlestick(candlestick_json, 'market_1')

        assert candlestick is not None
        assert isinstance(candlestick, Candlestick)
        assert candlestick.market_id == 'market_1'
        assert candlestick.open_price == 0.5
        assert candlestick.close_price == 0.52
        assert candlestick.volume == 1000

    def test_parse_single_candlestick_invalid(self):
        """Test parsing an invalid candlestick."""
        processor = KalshiDataProcessor()

        # Invalid prices (outside 0-1 range)
        invalid_candlestick = {
            'timestamp': '2024-01-15T13:00:00Z',
            'open': 1.5,  # Invalid
            'high': 2.0,  # Invalid
            'low': -0.1,  # Invalid
            'close': 1.2,  # Invalid
            'volume': 1000
        }

        candlestick = processor._parse_single_candlestick(invalid_candlestick, 'market_1')
        assert candlestick is None

    def test_calculate_technical_indicators(self, sample_kalshi_dataframe):
        """Test technical indicator calculation."""
        processor = KalshiDataProcessor()

        df_with_indicators = processor.calculate_technical_indicators(sample_kalshi_dataframe)

        # Check that indicators were added
        expected_indicators = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
            'price_change', 'price_momentum', 'volatility',
            'bid_ask_spread', 'volume_ma', 'volume_ratio'
        ]

        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns

        # Check RSI values are in valid range
        rsi_values = df_with_indicators['rsi'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        processor = KalshiDataProcessor()

        # Create sample price series with known pattern
        prices = pd.Series([0.5, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55, 0.54, 0.56,
                           0.55, 0.57, 0.56, 0.58, 0.57, 0.59, 0.58, 0.60])

        rsi = processor._calculate_rsi(prices, period=14)

        # RSI should be between 0 and 100
        rsi_values = rsi.dropna()
        assert all(0 <= val <= 100 for val in rsi_values)

    def test_resample_candlesticks(self, sample_kalshi_dataframe):
        """Test candlestick resampling."""
        processor = KalshiDataProcessor()

        # Resample to 5-minute intervals
        resampled_df = processor.resample_candlesticks(sample_kalshi_dataframe, '5T')

        assert isinstance(resampled_df, pd.DataFrame)
        assert len(resampled_df) < len(sample_kalshi_dataframe)  # Should be fewer records
        assert 'open_price' in resampled_df.columns
        assert 'close_price' in resampled_df.columns

    def test_detect_price_anomalies(self, sample_kalshi_dataframe):
        """Test price anomaly detection."""
        processor = KalshiDataProcessor()

        # Add some extreme values to create anomalies
        df_with_anomalies = sample_kalshi_dataframe.copy()
        df_with_anomalies.loc[5, 'close_price'] = 0.95  # Extreme high
        df_with_anomalies.loc[10, 'close_price'] = 0.05  # Extreme low

        df_with_flags = processor.detect_price_anomalies(df_with_anomalies, threshold=2.0)

        assert 'price_anomaly' in df_with_flags.columns
        assert 'price_z_score' in df_with_flags.columns
        assert 'volume_anomaly' in df_with_flags.columns

        # Should detect at least some anomalies
        assert df_with_flags['price_anomaly'].sum() > 0

    def test_to_dataframe(self, sample_kalshi_data):
        """Test conversion to DataFrame."""
        processor = KalshiDataProcessor()

        candlesticks = processor.parse_candlesticks(sample_kalshi_data)
        df = processor.to_dataframe(candlesticks)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(candlesticks)
        assert 'timestamp' in df.columns
        assert 'market_id' in df.columns
        assert 'close_price' in df.columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

        # Check sorted by timestamp
        assert df['timestamp'].is_monotonic_increasing

    def test_to_dataframe_empty(self):
        """Test DataFrame conversion with empty candlesticks."""
        processor = KalshiDataProcessor()

        df = processor.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_save_processed_data_parquet(self, sample_kalshi_dataframe, tmp_path):
        """Test saving data to Parquet format."""
        processor = KalshiDataProcessor()
        output_path = tmp_path / "test_output.parquet"

        processor.save_processed_data(sample_kalshi_dataframe, output_path)

        assert output_path.exists()

        # Verify data can be loaded back
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded_df, sample_kalshi_dataframe)

    def test_process_file_end_to_end(self, sample_kalshi_data, temp_json_file, tmp_path):
        """Test complete file processing workflow."""
        processor = KalshiDataProcessor()

        input_file = temp_json_file(sample_kalshi_data)
        output_file = tmp_path / "processed_output.parquet"

        df = processor.process_file(
            input_file, output_file,
            add_technical_indicators=True,
            detect_anomalies=True
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert output_file.exists()

        # Check required columns
        required_columns = ['timestamp', 'market_id', 'close_price', 'volume']
        for col in required_columns:
            assert col in df.columns

        # Check technical indicators were added
        assert 'sma_5' in df.columns
        assert 'rsi' in df.columns
        assert 'price_anomaly' in df.columns

    def test_process_file_with_resampling(self, sample_kalshi_data, temp_json_file, tmp_path):
        """Test file processing with resampling."""
        processor = KalshiDataProcessor()

        input_file = temp_json_file(sample_kalshi_data)
        output_file = tmp_path / "processed_output.parquet"

        df = processor.process_file(
            input_file, output_file,
            resample_frequency='5T'
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_process_directory(self, sample_kalshi_data, temp_directory, tmp_path):
        """Test processing multiple files in directory."""
        processor = KalshiDataProcessor()

        input_dir = temp_directory(['input'])
        output_dir = tmp_path / "output"

        # Create multiple JSON files
        for i in range(3):
            test_data = sample_kalshi_data.copy()
            test_data['market_id'] = f'market_{i}'

            json_file = input_dir / "input" / f"market_{i}.json"
            with open(json_file, 'w') as f:
                json.dump(test_data, f)

        dataframes = processor.process_directory(input_dir / "input", output_dir)

        assert len(dataframes) == 3
        assert all(isinstance(df, pd.DataFrame) for df in dataframes)
        assert len(list(output_dir.glob("*.parquet"))) == 3


class TestCandlestickValidator:
    """Test cases for CandlestickValidator."""

    def test_valid_candlestick(self):
        """Test validation of valid candlestick."""
        valid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'market_id': 'market_1',
            'open_price': 0.5,
            'high_price': 0.55,
            'low_price': 0.45,
            'close_price': 0.52,
            'volume': 1000
        }

        validator = CandlestickValidator(**valid_data)
        assert validator.market_id == 'market_1'
        assert validator.open_price == 0.5

    def test_invalid_price_range(self):
        """Test validation with prices outside valid range."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'market_id': 'market_1',
            'open_price': 1.5,  # Invalid
            'high_price': 1.6,
            'low_price': 0.45,
            'close_price': 1.52,
            'volume': 1000
        }

        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            CandlestickValidator(**invalid_data)

    def test_negative_volume(self):
        """Test validation with negative volume."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'market_id': 'market_1',
            'open_price': 0.5,
            'high_price': 0.55,
            'low_price': 0.45,
            'close_price': 0.52,
            'volume': -100  # Invalid
        }

        with pytest.raises(ValueError, match="Volume cannot be negative"):
            CandlestickValidator(**invalid_data)

    def test_high_less_than_low(self):
        """Test validation with high price less than low price."""
        invalid_data = {
            'timestamp': '2024-01-15T13:00:00Z',
            'market_id': 'market_1',
            'open_price': 0.5,
            'high_price': 0.4,  # Less than low_price
            'low_price': 0.45,
            'close_price': 0.42,
            'volume': 1000
        }

        with pytest.raises(ValueError, match="High price cannot be less than low price"):
            CandlestickValidator(**invalid_data)