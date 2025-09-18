"""Technical indicators for Kalshi price movement analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import get_config, get_logger


logger = get_logger(__name__)


class PriceRegime(Enum):
    """Price regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLUME = "low_volume"


class VolumeRegime(Enum):
    """Volume regime classifications."""
    HIGH_VOLUME = "high_volume"
    NORMAL_VOLUME = "normal_volume"
    LOW_VOLUME = "low_volume"
    VOLUME_SPIKE = "volume_spike"


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    signal_type: str
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: pd.Timestamp
    description: str


class TechnicalIndicators:
    """Comprehensive technical indicators for Kalshi price data."""

    def __init__(self, config=None):
        """Initialize technical indicators calculator.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.TechnicalIndicators")

        # Technical indicator parameters
        self.sma_periods = [5, 10, 20, 50]
        self.ema_periods = [5, 10, 20, 50]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.atr_period = 14
        self.volume_sma_period = 20
        
        # Thresholds
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.price_change_threshold = 0.02  # 2% price change

    def extract_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive technical indicator features.

        Args:
            price_data: DataFrame with OHLCV price data

        Returns:
            DataFrame with technical indicator features
        """
        try:
            self.logger.info(f"Extracting technical features from {len(price_data)} price points")
            
            # Validate input data
            price_data = self._validate_price_data(price_data)
            
            # Create features dataframe
            features_df = price_data.copy()
            
            # Add moving averages
            features_df = self._add_moving_averages(features_df)
            
            # Add momentum indicators
            features_df = self._add_momentum_indicators(features_df)
            
            # Add volatility indicators
            features_df = self._add_volatility_indicators(features_df)
            
            # Add volume indicators
            features_df = self._add_volume_indicators(features_df)
            
            # Add price pattern features
            features_df = self._add_price_patterns(features_df)
            
            # Add regime classification
            features_df = self._add_regime_classification(features_df)
            
            # Add relative strength features
            features_df = self._add_relative_strength(features_df)
            
            # Add microstructure features
            features_df = self._add_microstructure_features(features_df)
            
            self.logger.info(f"Extracted {len(features_df.columns)} technical features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            raise

    def _validate_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data."""
        required_cols = ['timestamp', 'close_price', 'volume']
        optional_cols = ['open_price', 'high_price', 'low_price', 'bid_price', 'ask_price']
        
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        price_data = price_data.copy()
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data = price_data.sort_values('timestamp').reset_index(drop=True)
        
        # Fill missing optional columns
        for col in optional_cols:
            if col not in price_data.columns:
                if 'price' in col:
                    price_data[col] = price_data['close_price']
                else:
                    price_data[col] = 0
        
        # Forward fill missing values
        price_data = price_data.fillna(method='ffill')
        
        return price_data

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving average indicators."""
        close_price = df['close_price']
        
        # Simple Moving Averages
        for period in self.sma_periods:
            df[f'sma_{period}'] = close_price.rolling(window=period).mean()
            
            # Price relative to SMA
            df[f'price_vs_sma_{period}'] = (close_price / df[f'sma_{period}'] - 1) * 100
            
            # SMA slope (rate of change)
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(periods=3)
        
        # Exponential Moving Averages
        for period in self.ema_periods:
            df[f'ema_{period}'] = close_price.ewm(span=period).mean()
            
            # Price relative to EMA
            df[f'price_vs_ema_{period}'] = (close_price / df[f'ema_{period}'] - 1) * 100
        
        # Moving average convergence/divergence patterns
        if len(self.sma_periods) >= 2:
            fast_sma = f'sma_{self.sma_periods[0]}'
            slow_sma = f'sma_{self.sma_periods[1]}'
            
            df['sma_convergence'] = df[fast_sma] - df[slow_sma]
            df['sma_crossover'] = (df['sma_convergence'] > 0).astype(int)
            df['sma_crossover_change'] = df['sma_crossover'].diff()
        
        # Volume-weighted moving average
        df['vwma_20'] = (close_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_vs_vwma'] = (close_price / df['vwma_20'] - 1) * 100
        
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        close_price = df['close_price']
        
        # Rate of Change (ROC)
        for period in [1, 3, 5, 10]:
            df[f'roc_{period}'] = close_price.pct_change(periods=period) * 100
        
        # Relative Strength Index (RSI)
        df['rsi'] = self._calculate_rsi(close_price, self.rsi_period)
        df['rsi_overbought'] = (df['rsi'] > self.rsi_overbought).astype(int)
        df['rsi_oversold'] = (df['rsi'] < self.rsi_oversold).astype(int)
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close_price)
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        df['macd_bullish'] = (macd_histogram > 0).astype(int)
        df['macd_crossover'] = (macd_line > macd_signal).astype(int)
        df['macd_crossover_change'] = df['macd_crossover'].diff()
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        df['stoch_oversold'] = ((df['stoch_k'] < 20) & (df['stoch_d'] < 20)).astype(int)
        df['stoch_overbought'] = ((df['stoch_k'] > 80) & (df['stoch_d'] > 80)).astype(int)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Money Flow Index (MFI)
        df['mfi'] = self._calculate_mfi(df)
        
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        close_price = df['close_price']
        high_price = df['high_price']
        low_price = df['low_price']
        
        # Average True Range (ATR)
        df['atr'] = self._calculate_atr(df, self.atr_period)
        df['atr_percent'] = df['atr'] / close_price * 100
        
        # Bollinger Bands
        bb_middle = close_price.rolling(self.bollinger_period).mean()
        bb_std = close_price.rolling(self.bollinger_period).std()
        
        df['bb_upper'] = bb_middle + (bb_std * self.bollinger_std)
        df['bb_lower'] = bb_middle - (bb_std * self.bollinger_std)
        df['bb_middle'] = bb_middle
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (close_price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band squeeze
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)
        
        # Historical volatility
        for period in [5, 10, 20]:
            returns = close_price.pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252) * 100
        
        # Volatility regime
        volatility_ma = df['volatility_20'].rolling(50).mean()
        df['volatility_regime'] = np.where(
            df['volatility_20'] > volatility_ma * 1.5, 'high',
            np.where(df['volatility_20'] < volatility_ma * 0.5, 'low', 'normal')
        )
        
        # High-Low spread
        df['hl_spread'] = (high_price - low_price) / close_price * 100
        df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
        
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        volume = df['volume']
        close_price = df['close_price']
        
        # Volume moving averages
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        df['volume_spike'] = (df['volume_ratio'] > self.volume_spike_threshold).astype(int)
        
        # Volume-Price Trend (VPT)
        price_change = close_price.pct_change()
        df['vpt'] = (volume * price_change).cumsum()
        
        # On-Balance Volume (OBV)
        price_direction = np.where(close_price > close_price.shift(1), 1, -1)
        df['obv'] = (volume * price_direction).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # Accumulation/Distribution Line
        df['ad_line'] = self._calculate_ad_line(df)
        
        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = self._calculate_vwap(df)
        df['price_vs_vwap'] = (close_price / df['vwap'] - 1) * 100
        
        # Volume Profile features
        df['volume_percentile'] = volume.rolling(50).rank(pct=True) * 100
        
        # Volume trend
        df['volume_trend'] = volume.rolling(10).mean() / volume.rolling(30).mean()
        
        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition features."""
        close_price = df['close_price']
        high_price = df['high_price']
        low_price = df['low_price']
        open_price = df['open_price']
        
        # Price gaps
        df['gap_up'] = (open_price > close_price.shift(1) * 1.01).astype(int)
        df['gap_down'] = (open_price < close_price.shift(1) * 0.99).astype(int)
        
        # Support and resistance levels
        df['local_high'] = (high_price == high_price.rolling(5, center=True).max()).astype(int)
        df['local_low'] = (low_price == low_price.rolling(5, center=True).min()).astype(int)
        
        # Higher highs, lower lows pattern
        df['higher_high'] = ((high_price > high_price.shift(1)) & 
                           (high_price.shift(1) > high_price.shift(2))).astype(int)
        df['lower_low'] = ((low_price < low_price.shift(1)) & 
                         (low_price.shift(1) < low_price.shift(2))).astype(int)
        
        # Candlestick patterns (simplified)
        body = abs(close_price - open_price)
        upper_shadow = high_price - np.maximum(close_price, open_price)
        lower_shadow = np.minimum(close_price, open_price) - low_price
        
        df['doji'] = (body / (high_price - low_price) < 0.1).astype(int)
        df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body)).astype(int)
        df['shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body)).astype(int)
        
        # Price momentum patterns
        df['bullish_momentum'] = (
            (close_price > close_price.shift(1)) &
            (close_price.shift(1) > close_price.shift(2)) &
            (close_price.shift(2) > close_price.shift(3))
        ).astype(int)
        
        df['bearish_momentum'] = (
            (close_price < close_price.shift(1)) &
            (close_price.shift(1) < close_price.shift(2)) &
            (close_price.shift(2) < close_price.shift(3))
        ).astype(int)
        
        return df

    def _add_regime_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features."""
        close_price = df['close_price']
        
        # Trend classification using multiple timeframes
        for period in [10, 20, 50]:
            sma = close_price.rolling(period).mean()
            df[f'trend_{period}'] = np.where(
                close_price > sma * 1.02, 'uptrend',
                np.where(close_price < sma * 0.98, 'downtrend', 'sideways')
            )
        
        # Volatility regime
        volatility = close_price.pct_change().rolling(20).std()
        volatility_percentile = volatility.rolling(100).rank(pct=True)
        
        df['volatility_regime'] = np.where(
            volatility_percentile > 0.8, 'high_vol',
            np.where(volatility_percentile < 0.2, 'low_vol', 'normal_vol')
        )
        
        # Volume regime
        volume_percentile = df['volume'].rolling(50).rank(pct=True)
        df['volume_regime'] = np.where(
            volume_percentile > 0.8, 'high_volume',
            np.where(volume_percentile < 0.2, 'low_volume', 'normal_volume')
        )
        
        # Combined regime
        df['market_regime'] = df['trend_20'] + '_' + df['volatility_regime'] + '_' + df['volume_regime']
        
        return df

    def _add_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength features."""
        close_price = df['close_price']
        
        # Price percentile ranks
        for period in [20, 50, 100]:
            df[f'price_percentile_{period}'] = close_price.rolling(period).rank(pct=True) * 100
        
        # Distance from recent high/low
        for period in [20, 50]:
            period_high = close_price.rolling(period).max()
            period_low = close_price.rolling(period).min()
            
            df[f'distance_from_high_{period}'] = (close_price / period_high - 1) * 100
            df[f'distance_from_low_{period}'] = (close_price / period_low - 1) * 100
        
        # Relative strength vs moving averages
        df['above_all_sma'] = (
            (close_price > df['sma_5']) &
            (close_price > df['sma_10']) &
            (close_price > df['sma_20'])
        ).astype(int)
        
        df['below_all_sma'] = (
            (close_price < df['sma_5']) &
            (close_price < df['sma_10']) &
            (close_price < df['sma_20'])
        ).astype(int)
        
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            # Bid-ask spread
            df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
            df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['close_price'] * 100
            df['spread_ma'] = df['bid_ask_spread_pct'].rolling(20).mean()
            df['spread_vs_ma'] = df['bid_ask_spread_pct'] / df['spread_ma']
            
            # Mid price
            df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
            df['price_vs_mid'] = (df['close_price'] / df['mid_price'] - 1) * 100
            
            # Spread volatility
            df['spread_volatility'] = df['bid_ask_spread_pct'].rolling(20).std()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Trading session features
        df['market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['after_hours'] = ((df['hour'] < 9) | (df['hour'] >= 16)).astype(int)
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD line, signal line, and histogram."""
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        high = df['high_price']
        low = df['low_price']
        close = df['close_price']
        
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high = df['high_price']
        low = df['low_price']
        close = df['close_price']
        
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(period).sum()
        negative_flow_sum = negative_flow.rolling(period).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high_price']
        low = df['low_price']
        close = df['close_price']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr

    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        high = df['high_price']
        low = df['low_price']
        close = df['close_price']
        volume = df['volume']
        
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        
        ad_volume = clv * volume
        ad_line = ad_volume.cumsum()
        
        return ad_line

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Weighted Average Price."""
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        
        # Reset VWAP daily (simplified - using 24-hour periods)
        df_copy = df.copy()
        df_copy['date'] = df_copy['timestamp'].dt.date
        
        vwap_list = []
        for date in df_copy['date'].unique():
            date_mask = df_copy['date'] == date
            date_data = df_copy[date_mask]
            
            cumulative_pv = (typical_price[date_mask] * df['volume'][date_mask]).cumsum()
            cumulative_volume = df['volume'][date_mask].cumsum()
            
            daily_vwap = cumulative_pv / cumulative_volume
            vwap_list.extend(daily_vwap.tolist())
        
        return pd.Series(vwap_list, index=df.index)

    def generate_signals(self, features_df: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate trading signals from technical indicators."""
        signals = []
        
        for idx, row in features_df.iterrows():
            timestamp = row['timestamp']
            
            # RSI signals
            if not pd.isna(row['rsi']):
                if row['rsi'] > self.rsi_overbought:
                    signals.append(TechnicalSignal(
                        signal_type="RSI_OVERBOUGHT",
                        strength=-0.7,
                        confidence=0.6,
                        timestamp=timestamp,
                        description=f"RSI overbought at {row['rsi']:.1f}"
                    ))
                elif row['rsi'] < self.rsi_oversold:
                    signals.append(TechnicalSignal(
                        signal_type="RSI_OVERSOLD",
                        strength=0.7,
                        confidence=0.6,
                        timestamp=timestamp,
                        description=f"RSI oversold at {row['rsi']:.1f}"
                    ))
            
            # MACD signals
            if not pd.isna(row['macd_crossover_change']):
                if row['macd_crossover_change'] == 1:
                    signals.append(TechnicalSignal(
                        signal_type="MACD_BULLISH_CROSSOVER",
                        strength=0.6,
                        confidence=0.7,
                        timestamp=timestamp,
                        description="MACD bullish crossover"
                    ))
                elif row['macd_crossover_change'] == -1:
                    signals.append(TechnicalSignal(
                        signal_type="MACD_BEARISH_CROSSOVER",
                        strength=-0.6,
                        confidence=0.7,
                        timestamp=timestamp,
                        description="MACD bearish crossover"
                    ))
            
            # Volume spike signals
            if not pd.isna(row['volume_spike']) and row['volume_spike'] == 1:
                price_change = row['roc_1'] if not pd.isna(row['roc_1']) else 0
                strength = np.sign(price_change) * min(abs(price_change) / 5, 1.0)
                
                signals.append(TechnicalSignal(
                    signal_type="VOLUME_SPIKE",
                    strength=strength,
                    confidence=0.5,
                    timestamp=timestamp,
                    description=f"Volume spike with {price_change:.1f}% price change"
                ))
            
            # Bollinger Band signals
            if not pd.isna(row['bb_position']):
                if row['bb_position'] > 0.95:
                    signals.append(TechnicalSignal(
                        signal_type="BB_UPPER_BREACH",
                        strength=-0.5,
                        confidence=0.4,
                        timestamp=timestamp,
                        description="Price near upper Bollinger Band"
                    ))
                elif row['bb_position'] < 0.05:
                    signals.append(TechnicalSignal(
                        signal_type="BB_LOWER_BREACH",
                        strength=0.5,
                        confidence=0.4,
                        timestamp=timestamp,
                        description="Price near lower Bollinger Band"
                    ))
        
        return signals

    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of feature names to their descriptions."""
        return {
            # Moving averages
            'sma_5': '5-period simple moving average',
            'sma_10': '10-period simple moving average',
            'sma_20': '20-period simple moving average',
            'price_vs_sma_20': 'Price deviation from 20-period SMA (%)',
            'ema_5': '5-period exponential moving average',
            'ema_20': '20-period exponential moving average',
            
            # Momentum indicators
            'rsi': 'Relative Strength Index (14-period)',
            'roc_1': '1-period rate of change (%)',
            'roc_5': '5-period rate of change (%)',
            'macd_line': 'MACD line',
            'macd_histogram': 'MACD histogram',
            'stoch_k': 'Stochastic %K',
            'williams_r': 'Williams %R',
            'mfi': 'Money Flow Index',
            
            # Volatility indicators
            'atr': 'Average True Range',
            'bb_width': 'Bollinger Bands width (%)',
            'bb_position': 'Position within Bollinger Bands (0-1)',
            'volatility_20': '20-period historical volatility (%)',
            'hl_spread': 'High-Low spread (%)',
            
            # Volume indicators
            'volume_ratio': 'Volume ratio vs 20-period average',
            'volume_spike': 'Volume spike indicator (1/0)',
            'obv': 'On-Balance Volume',
            'vpt': 'Volume-Price Trend',
            'ad_line': 'Accumulation/Distribution Line',
            'vwap': 'Volume-Weighted Average Price',
            'price_vs_vwap': 'Price deviation from VWAP (%)',
            
            # Pattern recognition
            'local_high': 'Local high indicator (1/0)',
            'local_low': 'Local low indicator (1/0)',
            'doji': 'Doji candlestick pattern (1/0)',
            'hammer': 'Hammer candlestick pattern (1/0)',
            'bullish_momentum': 'Bullish momentum pattern (1/0)',
            'bearish_momentum': 'Bearish momentum pattern (1/0)',
            
            # Regime classification
            'trend_20': 'Trend classification (20-period)',
            'volatility_regime': 'Volatility regime classification',
            'volume_regime': 'Volume regime classification',
            'market_regime': 'Combined market regime',
            
            # Relative strength
            'price_percentile_50': 'Price percentile rank (50-period)',
            'distance_from_high_20': 'Distance from 20-period high (%)',
            'distance_from_low_20': 'Distance from 20-period low (%)',
            'above_all_sma': 'Price above all SMAs (1/0)',
            'below_all_sma': 'Price below all SMAs (1/0)',
            
            # Microstructure
            'bid_ask_spread_pct': 'Bid-ask spread (%)',
            'spread_vs_ma': 'Spread vs moving average ratio',
            'price_vs_mid': 'Price vs mid-point deviation (%)'
        }