"""Kalshi data processor for price candlestick data."""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, validator

from ..config import get_config, get_logger


logger = get_logger(__name__)


@dataclass
class Candlestick:
    """Represents a price candlestick for a Kalshi market."""
    timestamp: datetime
    market_id: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    num_trades: Optional[int] = None
    vwap: Optional[float] = None


class CandlestickValidator(BaseModel):
    """Pydantic validator for candlestick data."""
    timestamp: datetime
    market_id: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    num_trades: Optional[int] = None
    vwap: Optional[float] = None

    @validator('timestamp')
    def validate_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @validator('open_price', 'high_price', 'low_price', 'close_price')
    def validate_prices(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Price must be between 0 and 1: {v}")
        return v

    @validator('volume')
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError(f"Volume cannot be negative: {v}")
        return v

    @validator('high_price')
    def validate_high_price(cls, v, values):
        if 'low_price' in values and v < values['low_price']:
            raise ValueError("High price cannot be less than low price")
        return v


class KalshiDataProcessor:
    """Processes Kalshi price candlestick data."""

    def __init__(self, config=None):
        """Initialize the Kalshi data processor.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.KalshiDataProcessor")

        # Technical indicators configuration
        self.technical_indicators = {
            'sma_periods': [5, 10, 20],
            'ema_periods': [5, 10, 20],
            'rsi_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2
        }

    def load_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load Kalshi candlestick data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Kalshi data file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Successfully loaded Kalshi data from {file_path}")
            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading Kalshi data file {file_path}: {e}")
            raise

    def parse_candlesticks(self, raw_data: Dict[str, Any]) -> List[Candlestick]:
        """Parse raw JSON data into structured candlestick objects.

        Args:
            raw_data: Raw JSON data from Kalshi API

        Returns:
            List of Candlestick objects

        Raises:
            ValueError: If data structure is invalid
        """
        try:
            candlesticks = []

            # Extract market metadata
            market_id = raw_data.get('market_id', 'unknown')

            # Process each candlestick
            candles_data = raw_data.get('candlesticks', [])

            for candle_json in candles_data:
                try:
                    candlestick = self._parse_single_candlestick(candle_json, market_id)
                    if candlestick:
                        candlesticks.append(candlestick)

                except Exception as e:
                    self.logger.warning(f"Failed to parse candlestick: {e}")
                    continue

            # Sort by timestamp
            candlesticks.sort(key=lambda x: x.timestamp)

            self.logger.info(f"Successfully parsed {len(candlesticks)} candlesticks for market {market_id}")
            return candlesticks

        except Exception as e:
            self.logger.error(f"Error parsing candlestick data: {e}")
            raise ValueError(f"Invalid candlestick data structure: {e}")

    def _parse_single_candlestick(
        self,
        candle_json: Dict[str, Any],
        market_id: str
    ) -> Optional[Candlestick]:
        """Parse a single candlestick from JSON data.

        Args:
            candle_json: Single candlestick JSON data
            market_id: Market identifier

        Returns:
            Candlestick object or None if invalid
        """
        try:
            # Validate required fields
            validator_data = CandlestickValidator(
                timestamp=candle_json['timestamp'],
                market_id=market_id,
                open_price=candle_json['open'],
                high_price=candle_json['high'],
                low_price=candle_json['low'],
                close_price=candle_json['close'],
                volume=candle_json.get('volume', 0),
                bid_price=candle_json.get('bid'),
                ask_price=candle_json.get('ask'),
                bid_size=candle_json.get('bid_size'),
                ask_size=candle_json.get('ask_size'),
                num_trades=candle_json.get('trades'),
                vwap=candle_json.get('vwap')
            )

            return Candlestick(**validator_data.dict())

        except Exception as e:
            self.logger.debug(f"Failed to parse candlestick: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()

        try:
            # Simple Moving Averages
            for period in self.technical_indicators['sma_periods']:
                df[f'sma_{period}'] = df['close_price'].rolling(window=period).mean()

            # Exponential Moving Averages
            for period in self.technical_indicators['ema_periods']:
                df[f'ema_{period}'] = df['close_price'].ewm(span=period).mean()

            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close_price'], self.technical_indicators['rsi_period'])

            # Bollinger Bands
            bb_period = self.technical_indicators['bollinger_period']
            bb_std = self.technical_indicators['bollinger_std']

            bb_mean = df['close_price'].rolling(window=bb_period).mean()
            bb_std_dev = df['close_price'].rolling(window=bb_period).std()

            df['bb_upper'] = bb_mean + (bb_std_dev * bb_std)
            df['bb_lower'] = bb_mean - (bb_std_dev * bb_std)
            df['bb_middle'] = bb_mean

            # Price momentum and volatility
            df['price_change'] = df['close_price'].pct_change()
            df['price_momentum'] = df['close_price'].pct_change(periods=5)
            df['volatility'] = df['price_change'].rolling(window=20).std()

            # Bid-Ask Spread
            df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
            df['bid_ask_spread_pct'] = df['bid_ask_spread'] / df['close_price']

            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # VWAP deviation
            df['vwap_deviation'] = (df['close_price'] - df['vwap']) / df['vwap']

            self.logger.info("Successfully calculated technical indicators")

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            prices: Series of prices
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def resample_candlesticks(
        self,
        df: pd.DataFrame,
        frequency: str = '60S'
    ) -> pd.DataFrame:
        """Resample candlesticks to different time frequencies.

        Args:
            df: DataFrame with candlestick data
            frequency: Target frequency (e.g., '60S', '5T', '1H')

        Returns:
            Resampled DataFrame
        """
        try:
            # Ensure timestamp is datetime and set as index
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Resample OHLCV data
            resampled = df.resample(frequency).agg({
                'open_price': 'first',
                'high_price': 'max',
                'low_price': 'min',
                'close_price': 'last',
                'volume': 'sum',
                'bid_price': 'last',
                'ask_price': 'last',
                'bid_size': 'last',
                'ask_size': 'last',
                'num_trades': 'sum',
                'vwap': 'mean'
            })

            # Forward fill missing values
            resampled.fillna(method='ffill', inplace=True)

            # Reset index
            resampled.reset_index(inplace=True)

            # Add market_id back
            if 'market_id' in df.columns:
                resampled['market_id'] = df['market_id'].iloc[0]

            self.logger.info(f"Resampled data to {frequency} frequency: {len(resampled)} periods")
            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling candlesticks: {e}")
            raise

    def detect_price_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Detect price anomalies using statistical methods.

        Args:
            df: DataFrame with price data
            threshold: Z-score threshold for anomaly detection

        Returns:
            DataFrame with anomaly flags
        """
        df = df.copy()

        try:
            # Calculate z-scores for price changes
            price_changes = df['close_price'].pct_change()
            mean_change = price_changes.mean()
            std_change = price_changes.std()

            z_scores = (price_changes - mean_change) / std_change
            df['price_anomaly'] = np.abs(z_scores) > threshold
            df['price_z_score'] = z_scores

            # Volume anomalies
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            volume_z_scores = (df['volume'] - volume_mean) / volume_std

            df['volume_anomaly'] = np.abs(volume_z_scores) > threshold
            df['volume_z_score'] = volume_z_scores

            # Spread anomalies
            if 'bid_ask_spread' in df.columns:
                spread_mean = df['bid_ask_spread'].mean()
                spread_std = df['bid_ask_spread'].std()
                spread_z_scores = (df['bid_ask_spread'] - spread_mean) / spread_std

                df['spread_anomaly'] = np.abs(spread_z_scores) > threshold
                df['spread_z_score'] = spread_z_scores

            anomaly_count = df['price_anomaly'].sum()
            self.logger.info(f"Detected {anomaly_count} price anomalies")

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            raise

        return df

    def to_dataframe(self, candlesticks: List[Candlestick]) -> pd.DataFrame:
        """Convert candlesticks to pandas DataFrame.

        Args:
            candlesticks: List of Candlestick objects

        Returns:
            Pandas DataFrame
        """
        if not candlesticks:
            return pd.DataFrame()

        # Convert to dictionaries
        candle_dicts = []
        for candle in candlesticks:
            candle_dict = {
                'timestamp': candle.timestamp,
                'market_id': candle.market_id,
                'open_price': candle.open_price,
                'high_price': candle.high_price,
                'low_price': candle.low_price,
                'close_price': candle.close_price,
                'volume': candle.volume,
                'bid_price': candle.bid_price,
                'ask_price': candle.ask_price,
                'bid_size': candle.bid_size,
                'ask_size': candle.ask_size,
                'num_trades': candle.num_trades,
                'vwap': candle.vwap
            }
            candle_dicts.append(candle_dict)

        df = pd.DataFrame(candle_dicts)

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.info(f"Created DataFrame with {len(df)} candlesticks")
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path]):
        """Save processed data to file.

        Args:
            df: Processed DataFrame
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            df.to_json(output_path, orient='records', date_format='iso', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        self.logger.info(f"Saved processed Kalshi data to {output_path}")

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        add_technical_indicators: bool = True,
        resample_frequency: Optional[str] = None,
        detect_anomalies: bool = True
    ) -> pd.DataFrame:
        """Process a single Kalshi data file end-to-end.

        Args:
            input_path: Input JSON file path
            output_path: Output file path
            add_technical_indicators: Whether to add technical indicators
            resample_frequency: Frequency for resampling (optional)
            detect_anomalies: Whether to detect anomalies

        Returns:
            Processed DataFrame
        """
        try:
            # Load and parse data
            raw_data = self.load_json_file(input_path)
            candlesticks = self.parse_candlesticks(raw_data)

            # Convert to DataFrame
            df = self.to_dataframe(candlesticks)

            # Resample if requested
            if resample_frequency:
                df = self.resample_candlesticks(df, resample_frequency)

            # Add technical indicators
            if add_technical_indicators:
                df = self.calculate_technical_indicators(df)

            # Detect anomalies
            if detect_anomalies:
                df = self.detect_price_anomalies(df)

            # Save processed data
            self.save_processed_data(df, output_path)

            return df

        except Exception as e:
            self.logger.error(f"Failed to process Kalshi file {input_path}: {e}")
            raise

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[pd.DataFrame]:
        """Process all JSON files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            **kwargs: Additional arguments for process_file

        Returns:
            List of processed DataFrames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(input_dir.glob('*.json'))

        if not json_files:
            self.logger.warning(f"No JSON files found in {input_dir}")
            return []

        dataframes = []

        for json_file in json_files:
            try:
                output_file = output_dir / f"{json_file.stem}_processed.parquet"
                df = self.process_file(json_file, output_file, **kwargs)
                dataframes.append(df)

            except Exception as e:
                self.logger.error(f"Failed to process {json_file}: {e}")
                continue

        self.logger.info(f"Successfully processed {len(dataframes)}/{len(json_files)} Kalshi data files")
        return dataframes