"""Tests for TechnicalIndicators class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.nfl_trading.features.technical_indicators import TechnicalIndicators, TechnicalSignal


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators."""

    @pytest.fixture
    def indicators(self):
        """Create TechnicalIndicators instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        base_time = datetime.now()
        n_points = 100
        
        # Generate realistic price data
        prices = 50 + np.cumsum(np.random.randn(n_points) * 0.02)
        volumes = np.random.randint(1000, 10000, n_points)
        
        data = {
            'timestamp': [base_time + timedelta(minutes=i) for i in range(n_points)],
            'close_price': prices,
            'open_price': prices + np.random.randn(n_points) * 0.01,
            'high_price': prices + np.abs(np.random.randn(n_points) * 0.02),
            'low_price': prices - np.abs(np.random.randn(n_points) * 0.02),
            'volume': volumes,
            'bid_price': prices - 0.01,
            'ask_price': prices + 0.01
        }
        
        # Ensure high >= low and close between them
        df = pd.DataFrame(data)
        df['high_price'] = np.maximum(df['high_price'], df[['open_price', 'close_price']].max(axis=1))
        df['low_price'] = np.minimum(df['low_price'], df[['open_price', 'close_price']].min(axis=1))
        
        return df

    def test_initialization(self, indicators):
        """Test TechnicalIndicators initialization."""
        assert indicators.sma_periods == [5, 10, 20, 50]
        assert indicators.ema_periods == [5, 10, 20, 50]
        assert indicators.rsi_period == 14
        assert indicators.bollinger_period == 20
        assert indicators.rsi_overbought == 70
        assert indicators.rsi_oversold == 30

    def test_extract_features_basic(self, indicators, sample_price_data):
        """Test basic feature extraction."""
        features_df = indicators.extract_features(sample_price_data)
        
        # Check that features were extracted
        assert len(features_df) == len(sample_price_data)
        assert 'sma_5' in features_df.columns
        assert 'ema_10' in features_df.columns
        assert 'rsi' in features_df.columns
        assert 'bb_upper' in features_df.columns

    def test_validate_price_data(self, indicators):
        """Test price data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close_price': [50.0],
            'volume': [1000]
        })
        
        validated = indicators._validate_price_data(valid_data)
        assert len(validated) == 1
        assert 'open_price' in validated.columns
        assert 'high_price' in validated.columns
        
        # Missing required columns
        invalid_data = pd.DataFrame({'timestamp': [datetime.now()]})
        with pytest.raises(ValueError, match="Missing required columns"):
            indicators._validate_price_data(invalid_data)

    def test_moving_averages(self, indicators, sample_price_data):
        """Test moving average calculations."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_moving_averages(df)
        
        # Check SMA columns
        for period in indicators.sma_periods:
            sma_col = f'sma_{period}'
            assert sma_col in df.columns
            
            # SMA should be rolling mean
            expected_sma = df['close_price'].rolling(window=period).mean()
            pd.testing.assert_series_equal(df[sma_col], expected_sma, check_names=False)
        
        # Check EMA columns
        for period in indicators.ema_periods:
            ema_col = f'ema_{period}'
            assert ema_col in df.columns
            
            # EMA should be exponential weighted mean
            expected_ema = df['close_price'].ewm(span=period).mean()
            pd.testing.assert_series_equal(df[ema_col], expected_ema, check_names=False)
        
        # Check relative price features
        assert 'price_vs_sma_5' in df.columns
        assert 'price_vs_ema_5' in df.columns

    def test_momentum_indicators(self, indicators, sample_price_data):
        """Test momentum indicator calculations."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_momentum_indicators(df)
        
        # Check ROC columns
        roc_periods = [1, 3, 5, 10]
        for period in roc_periods:
            roc_col = f'roc_{period}'
            assert roc_col in df.columns
            
            # ROC should be percent change
            expected_roc = df['close_price'].pct_change(periods=period) * 100
            pd.testing.assert_series_equal(df[roc_col], expected_roc, check_names=False)
        
        # Check RSI
        assert 'rsi' in df.columns
        assert df['rsi'].min() >= 0
        assert df['rsi'].max() <= 100
        
        # Check MACD
        assert 'macd_line' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_histogram' in df.columns
        
        # Check Stochastic
        assert 'stoch_k' in df.columns
        assert 'stoch_d' in df.columns

    def test_rsi_calculation(self, indicators):
        """Test RSI calculation with known values."""
        # Create simple test data
        prices = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 
                           46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.8, 46.8, 46.8])
        
        rsi = indicators._calculate_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        
        # RSI should not be NaN after enough periods
        assert not rsi.iloc[-1:].isna().any()

    def test_macd_calculation(self, indicators):
        """Test MACD calculation."""
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        
        macd_line, macd_signal, macd_histogram = indicators._calculate_macd(prices)
        
        # MACD histogram should equal line - signal
        expected_histogram = macd_line - macd_signal
        pd.testing.assert_series_equal(macd_histogram, expected_histogram, check_names=False)
        
        # Check lengths
        assert len(macd_line) == len(prices)
        assert len(macd_signal) == len(prices)
        assert len(macd_histogram) == len(prices)

    def test_volatility_indicators(self, indicators, sample_price_data):
        """Test volatility indicator calculations."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_volatility_indicators(df)
        
        # Check ATR
        assert 'atr' in df.columns
        assert 'atr_percent' in df.columns
        
        # Check Bollinger Bands
        assert 'bb_upper' in df.columns
        assert 'bb_lower' in df.columns
        assert 'bb_middle' in df.columns
        assert 'bb_width' in df.columns
        
        # Upper band should be > lower band
        valid_rows = ~(df['bb_upper'].isna() | df['bb_lower'].isna())
        assert (df.loc[valid_rows, 'bb_upper'] > df.loc[valid_rows, 'bb_lower']).all()
        
        # Check volatility calculations
        assert 'volatility_20' in df.columns
        assert df['volatility_20'].min() >= 0  # Volatility should be non-negative

    def test_volume_indicators(self, indicators, sample_price_data):
        """Test volume indicator calculations."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_volume_indicators(df)
        
        # Check volume features
        assert 'volume_sma_20' in df.columns
        assert 'volume_ratio' in df.columns
        assert 'obv' in df.columns
        assert 'vwap' in df.columns
        
        # Volume ratio should be positive
        assert df['volume_ratio'].min() >= 0
        
        # OBV should be cumulative
        assert 'obv' in df.columns

    def test_atr_calculation(self, indicators, sample_price_data):
        """Test ATR calculation."""
        df = indicators._validate_price_data(sample_price_data)
        atr = indicators._calculate_atr(df, 14)
        
        # ATR should be non-negative
        assert atr.min() >= 0
        
        # ATR should not be NaN after enough periods
        assert not atr.iloc[14:].isna().any()

    def test_pattern_recognition(self, indicators, sample_price_data):
        """Test price pattern recognition."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_price_patterns(df)
        
        # Check pattern columns
        pattern_cols = ['doji', 'hammer', 'shooting_star', 'bullish_momentum', 'bearish_momentum']
        for col in pattern_cols:
            assert col in df.columns
            # Pattern columns should be binary
            assert df[col].isin([0, 1]).all()
        
        # Check gap detection
        assert 'gap_up' in df.columns
        assert 'gap_down' in df.columns

    def test_regime_classification(self, indicators, sample_price_data):
        """Test market regime classification."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_regime_classification(df)
        
        # Check trend classifications
        trend_cols = ['trend_10', 'trend_20', 'trend_50']
        for col in trend_cols:
            assert col in df.columns
            valid_trends = ['uptrend', 'downtrend', 'sideways']
            assert df[col].isin(valid_trends + [np.nan]).all()
        
        # Check volatility regime
        assert 'volatility_regime' in df.columns
        valid_vol_regimes = ['high_vol', 'low_vol', 'normal_vol']
        assert df['volatility_regime'].isin(valid_vol_regimes + [np.nan]).all()

    def test_signal_generation(self, indicators, sample_price_data):
        """Test trading signal generation."""
        features_df = indicators.extract_features(sample_price_data)
        signals = indicators.generate_signals(features_df)
        
        # Should return list of TechnicalSignal objects
        assert isinstance(signals, list)
        
        if signals:  # If any signals were generated
            signal = signals[0]
            assert isinstance(signal, TechnicalSignal)
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'strength')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'timestamp')
            
            # Strength and confidence should be in valid ranges
            assert -1.0 <= signal.strength <= 1.0
            assert 0.0 <= signal.confidence <= 1.0

    def test_microstructure_features(self, indicators, sample_price_data):
        """Test market microstructure features."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_microstructure_features(df)
        
        # Check bid-ask spread features
        assert 'bid_ask_spread' in df.columns
        assert 'bid_ask_spread_pct' in df.columns
        assert 'mid_price' in df.columns
        
        # Spread should be non-negative
        assert df['bid_ask_spread'].min() >= 0
        
        # Mid price should be between bid and ask
        mid_prices = df['mid_price'].dropna()
        bid_prices = df['bid_price'].dropna()
        ask_prices = df['ask_price'].dropna()
        
        if len(mid_prices) > 0:
            min_len = min(len(mid_prices), len(bid_prices), len(ask_prices))
            assert (mid_prices.iloc[:min_len] >= bid_prices.iloc[:min_len]).all()
            assert (mid_prices.iloc[:min_len] <= ask_prices.iloc[:min_len]).all()

    def test_edge_cases(self, indicators):
        """Test edge cases and error conditions."""
        # Test with constant prices
        constant_data = pd.DataFrame({
            'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(50)],
            'close_price': [50.0] * 50,
            'volume': [1000] * 50
        })
        
        features_df = indicators.extract_features(constant_data)
        
        # Should handle constant prices without errors
        assert len(features_df) == 50
        
        # RSI should handle constant prices
        assert 'rsi' in features_df.columns

    def test_missing_data_handling(self, indicators, sample_price_data):
        """Test handling of missing data."""
        # Introduce some NaN values
        data_with_nans = sample_price_data.copy()
        data_with_nans.loc[10:15, 'close_price'] = np.nan
        data_with_nans.loc[20:25, 'volume'] = np.nan
        
        features_df = indicators.extract_features(data_with_nans)
        
        # Should handle NaN values gracefully
        assert len(features_df) == len(data_with_nans)

    def test_feature_importance_mapping(self, indicators):
        """Test feature importance mapping."""
        mapping = indicators.get_feature_importance_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check some expected mappings
        expected_features = [
            'sma_20',
            'rsi',
            'macd_line',
            'bb_width',
            'volume_ratio'
        ]
        
        for feature in expected_features:
            assert feature in mapping
            assert isinstance(mapping[feature], str)
            assert len(mapping[feature]) > 0

    def test_performance_with_large_dataset(self, indicators):
        """Test performance with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        base_time = datetime.now()
        n_points = 1000
        
        prices = 50 + np.cumsum(np.random.randn(n_points) * 0.02)
        large_data = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i) for i in range(n_points)],
            'close_price': prices,
            'open_price': prices + np.random.randn(n_points) * 0.01,
            'high_price': prices + np.abs(np.random.randn(n_points) * 0.02),
            'low_price': prices - np.abs(np.random.randn(n_points) * 0.02),
            'volume': np.random.randint(1000, 10000, n_points)
        })
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        features_df = indicators.extract_features(large_data)
        end_time = time.time()
        
        assert end_time - start_time < 15  # Should complete within 15 seconds
        assert len(features_df) == n_points

    def test_bollinger_bands_properties(self, indicators, sample_price_data):
        """Test Bollinger Bands mathematical properties."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_volatility_indicators(df)
        
        # BB middle should equal SMA
        bb_middle = df['close_price'].rolling(indicators.bollinger_period).mean()
        pd.testing.assert_series_equal(df['bb_middle'], bb_middle, check_names=False)
        
        # Price position should be between 0 and 1 (mostly)
        bb_position = df['bb_position'].dropna()
        # Allow some outliers but most should be in range
        in_range_pct = ((bb_position >= 0) & (bb_position <= 1)).mean()
        assert in_range_pct > 0.8  # At least 80% should be in normal range

    def test_volume_indicators_properties(self, indicators, sample_price_data):
        """Test volume indicator mathematical properties."""
        df = indicators._validate_price_data(sample_price_data)
        df = indicators._add_volume_indicators(df)
        
        # VWAP should be reasonable relative to price
        vwap_values = df['vwap'].dropna()
        price_values = df['close_price'].dropna()
        
        if len(vwap_values) > 0 and len(price_values) > 0:
            # VWAP should be in reasonable range of prices
            price_min, price_max = price_values.min(), price_values.max()
            assert vwap_values.min() >= price_min * 0.5
            assert vwap_values.max() <= price_max * 1.5

    def test_technical_signal_properties(self, indicators):
        """Test technical signal object properties."""
        timestamp = datetime.now()
        signal = TechnicalSignal(
            signal_type="TEST_SIGNAL",
            strength=0.5,
            confidence=0.8,
            timestamp=timestamp,
            description="Test signal"
        )
        
        assert signal.signal_type == "TEST_SIGNAL"
        assert signal.strength == 0.5
        assert signal.confidence == 0.8
        assert signal.timestamp == timestamp
        assert signal.description == "Test signal"