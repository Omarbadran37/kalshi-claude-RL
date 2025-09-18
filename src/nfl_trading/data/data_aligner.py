"""Data aligner for synchronizing NFL play events with Kalshi price data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import get_config, get_logger


logger = get_logger(__name__)


class AlignmentMethod(Enum):
    """Alignment methods for synchronizing data."""
    NEAREST = "nearest"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATION = "interpolation"


@dataclass
class AlignmentResult:
    """Result of data alignment operation."""
    aligned_data: pd.DataFrame
    alignment_stats: Dict[str, Any]
    unmatched_plays: pd.DataFrame
    unmatched_prices: pd.DataFrame


class DataAligner:
    """Aligns NFL play-by-play data with Kalshi price data by timestamp."""

    def __init__(self, config=None):
        """Initialize the data aligner.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.DataAligner")

        # Alignment configuration
        self.time_tolerance = timedelta(seconds=self.config.processing.time_alignment_tolerance)
        self.default_alignment_method = AlignmentMethod.NEAREST

    def align_data(
        self,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame,
        method: AlignmentMethod = None,
        time_tolerance: Optional[timedelta] = None,
        price_columns: Optional[List[str]] = None
    ) -> AlignmentResult:
        """Align NFL play data with price data by timestamp.

        Args:
            nfl_data: DataFrame with NFL play-by-play data
            price_data: DataFrame with Kalshi price data
            method: Alignment method to use
            time_tolerance: Maximum time difference for matching
            price_columns: Specific price columns to align (optional)

        Returns:
            AlignmentResult object with aligned data and statistics
        """
        method = method or self.default_alignment_method
        time_tolerance = time_tolerance or self.time_tolerance

        try:
            self.logger.info(f"Aligning {len(nfl_data)} NFL plays with {len(price_data)} price points")

            # Validate input data
            nfl_data = self._validate_nfl_data(nfl_data)
            price_data = self._validate_price_data(price_data)

            # Determine price columns to include
            if price_columns is None:
                price_columns = self._get_default_price_columns(price_data)

            # Perform alignment based on method
            if method == AlignmentMethod.NEAREST:
                result = self._align_nearest(nfl_data, price_data, time_tolerance, price_columns)
            elif method == AlignmentMethod.FORWARD_FILL:
                result = self._align_forward_fill(nfl_data, price_data, time_tolerance, price_columns)
            elif method == AlignmentMethod.BACKWARD_FILL:
                result = self._align_backward_fill(nfl_data, price_data, time_tolerance, price_columns)
            elif method == AlignmentMethod.INTERPOLATION:
                result = self._align_interpolation(nfl_data, price_data, time_tolerance, price_columns)
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            # Calculate alignment statistics
            stats = self._calculate_alignment_stats(result.aligned_data, nfl_data, price_data)
            result.alignment_stats = stats

            self.logger.info(f"Successfully aligned data: {len(result.aligned_data)} matched records")
            return result

        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            raise

    def _validate_nfl_data(self, nfl_data: pd.DataFrame) -> pd.DataFrame:
        """Validate NFL data format and ensure timestamp column exists.

        Args:
            nfl_data: NFL DataFrame

        Returns:
            Validated DataFrame
        """
        required_columns = ['timestamp']
        missing_columns = [col for col in required_columns if col not in nfl_data.columns]

        if missing_columns:
            raise ValueError(f"NFL data missing required columns: {missing_columns}")

        # Ensure timestamp is datetime
        nfl_data = nfl_data.copy()
        nfl_data['timestamp'] = pd.to_datetime(nfl_data['timestamp'])

        # Sort by timestamp
        nfl_data = nfl_data.sort_values('timestamp').reset_index(drop=True)

        return nfl_data

    def _validate_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Validate price data format and ensure timestamp column exists.

        Args:
            price_data: Price DataFrame

        Returns:
            Validated DataFrame
        """
        required_columns = ['timestamp']
        missing_columns = [col for col in required_columns if col not in price_data.columns]

        if missing_columns:
            raise ValueError(f"Price data missing required columns: {missing_columns}")

        # Ensure timestamp is datetime
        price_data = price_data.copy()
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

        # Sort by timestamp
        price_data = price_data.sort_values('timestamp').reset_index(drop=True)

        return price_data

    def _get_default_price_columns(self, price_data: pd.DataFrame) -> List[str]:
        """Get default price columns to include in alignment.

        Args:
            price_data: Price DataFrame

        Returns:
            List of column names
        """
        default_columns = [
            'close_price', 'open_price', 'high_price', 'low_price', 'volume',
            'bid_price', 'ask_price', 'bid_ask_spread', 'vwap'
        ]

        return [col for col in default_columns if col in price_data.columns]

    def _align_nearest(
        self,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame,
        time_tolerance: timedelta,
        price_columns: List[str]
    ) -> AlignmentResult:
        """Align data using nearest timestamp matching.

        Args:
            nfl_data: NFL DataFrame
            price_data: Price DataFrame
            time_tolerance: Maximum time difference for matching
            price_columns: Price columns to include

        Returns:
            AlignmentResult
        """
        aligned_records = []
        unmatched_plays = []
        used_price_indices = set()

        for _, play in nfl_data.iterrows():
            play_time = play['timestamp']

            # Find nearest price point within tolerance
            time_diffs = (price_data['timestamp'] - play_time).abs()
            min_diff_idx = time_diffs.idxmin()
            min_diff = time_diffs.iloc[min_diff_idx]

            if min_diff <= time_tolerance:
                # Match found
                price_row = price_data.iloc[min_diff_idx]
                aligned_record = self._create_aligned_record(play, price_row, price_columns, min_diff)
                aligned_records.append(aligned_record)
                used_price_indices.add(min_diff_idx)
            else:
                # No match within tolerance
                unmatched_plays.append(play.to_dict())

        # Identify unmatched price data
        unmatched_price_indices = set(price_data.index) - used_price_indices
        unmatched_prices = price_data.loc[list(unmatched_price_indices)]

        return AlignmentResult(
            aligned_data=pd.DataFrame(aligned_records),
            alignment_stats={},
            unmatched_plays=pd.DataFrame(unmatched_plays),
            unmatched_prices=unmatched_prices
        )

    def _align_forward_fill(
        self,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame,
        time_tolerance: timedelta,
        price_columns: List[str]
    ) -> AlignmentResult:
        """Align data using forward fill method.

        Args:
            nfl_data: NFL DataFrame
            price_data: Price DataFrame
            time_tolerance: Maximum time difference for matching
            price_columns: Price columns to include

        Returns:
            AlignmentResult
        """
        aligned_records = []
        unmatched_plays = []

        for _, play in nfl_data.iterrows():
            play_time = play['timestamp']

            # Find last price point before the play
            earlier_prices = price_data[price_data['timestamp'] <= play_time]

            if not earlier_prices.empty:
                latest_price = earlier_prices.iloc[-1]
                time_diff = play_time - latest_price['timestamp']

                if time_diff <= time_tolerance:
                    aligned_record = self._create_aligned_record(play, latest_price, price_columns, time_diff)
                    aligned_records.append(aligned_record)
                else:
                    unmatched_plays.append(play.to_dict())
            else:
                unmatched_plays.append(play.to_dict())

        return AlignmentResult(
            aligned_data=pd.DataFrame(aligned_records),
            alignment_stats={},
            unmatched_plays=pd.DataFrame(unmatched_plays),
            unmatched_prices=pd.DataFrame()  # Will be calculated later
        )

    def _align_backward_fill(
        self,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame,
        time_tolerance: timedelta,
        price_columns: List[str]
    ) -> AlignmentResult:
        """Align data using backward fill method.

        Args:
            nfl_data: NFL DataFrame
            price_data: Price DataFrame
            time_tolerance: Maximum time difference for matching
            price_columns: Price columns to include

        Returns:
            AlignmentResult
        """
        aligned_records = []
        unmatched_plays = []

        for _, play in nfl_data.iterrows():
            play_time = play['timestamp']

            # Find first price point after the play
            later_prices = price_data[price_data['timestamp'] >= play_time]

            if not later_prices.empty:
                earliest_price = later_prices.iloc[0]
                time_diff = earliest_price['timestamp'] - play_time

                if time_diff <= time_tolerance:
                    aligned_record = self._create_aligned_record(play, earliest_price, price_columns, time_diff)
                    aligned_records.append(aligned_record)
                else:
                    unmatched_plays.append(play.to_dict())
            else:
                unmatched_plays.append(play.to_dict())

        return AlignmentResult(
            aligned_data=pd.DataFrame(aligned_records),
            alignment_stats={},
            unmatched_plays=pd.DataFrame(unmatched_plays),
            unmatched_prices=pd.DataFrame()  # Will be calculated later
        )

    def _align_interpolation(
        self,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame,
        time_tolerance: timedelta,
        price_columns: List[str]
    ) -> AlignmentResult:
        """Align data using linear interpolation.

        Args:
            nfl_data: NFL DataFrame
            price_data: Price DataFrame
            time_tolerance: Maximum time difference for matching
            price_columns: Price columns to include

        Returns:
            AlignmentResult
        """
        aligned_records = []
        unmatched_plays = []

        # Numeric price columns for interpolation
        numeric_price_columns = [col for col in price_columns
                               if pd.api.types.is_numeric_dtype(price_data[col])]

        for _, play in nfl_data.iterrows():
            play_time = play['timestamp']

            # Find surrounding price points
            before_prices = price_data[price_data['timestamp'] <= play_time]
            after_prices = price_data[price_data['timestamp'] >= play_time]

            if not before_prices.empty and not after_prices.empty:
                before_price = before_prices.iloc[-1]
                after_price = after_prices.iloc[0]

                # Check if within tolerance
                time_to_before = play_time - before_price['timestamp']
                time_to_after = after_price['timestamp'] - play_time

                if time_to_before <= time_tolerance or time_to_after <= time_tolerance:
                    # Perform interpolation
                    interpolated_data = self._interpolate_price_data(
                        before_price, after_price, play_time, numeric_price_columns
                    )

                    # Create aligned record with interpolated data
                    aligned_record = play.to_dict()
                    aligned_record.update(interpolated_data)
                    aligned_record['time_diff'] = min(time_to_before, time_to_after)
                    aligned_records.append(aligned_record)
                else:
                    unmatched_plays.append(play.to_dict())
            else:
                unmatched_plays.append(play.to_dict())

        return AlignmentResult(
            aligned_data=pd.DataFrame(aligned_records),
            alignment_stats={},
            unmatched_plays=pd.DataFrame(unmatched_plays),
            unmatched_prices=pd.DataFrame()  # Will be calculated later
        )

    def _interpolate_price_data(
        self,
        before_price: pd.Series,
        after_price: pd.Series,
        target_time: datetime,
        numeric_columns: List[str]
    ) -> Dict[str, Any]:
        """Linearly interpolate price data between two timestamps.

        Args:
            before_price: Price data before target time
            after_price: Price data after target time
            target_time: Target timestamp
            numeric_columns: Columns to interpolate

        Returns:
            Interpolated price data
        """
        interpolated = {}

        # Time-based interpolation weight
        total_time_diff = after_price['timestamp'] - before_price['timestamp']
        if total_time_diff.total_seconds() == 0:
            weight = 0.5
        else:
            elapsed_time = target_time - before_price['timestamp']
            weight = elapsed_time.total_seconds() / total_time_diff.total_seconds()

        # Interpolate numeric columns
        for col in numeric_columns:
            if pd.notna(before_price[col]) and pd.notna(after_price[col]):
                interpolated[col] = before_price[col] + weight * (after_price[col] - before_price[col])
            elif pd.notna(before_price[col]):
                interpolated[col] = before_price[col]
            elif pd.notna(after_price[col]):
                interpolated[col] = after_price[col]
            else:
                interpolated[col] = None

        # Use most recent non-numeric data
        non_numeric_columns = [col for col in before_price.index
                              if col not in numeric_columns and col != 'timestamp']

        for col in non_numeric_columns:
            interpolated[col] = before_price[col] if weight < 0.5 else after_price[col]

        return interpolated

    def _create_aligned_record(
        self,
        play: pd.Series,
        price: pd.Series,
        price_columns: List[str],
        time_diff: timedelta
    ) -> Dict[str, Any]:
        """Create an aligned record combining play and price data.

        Args:
            play: NFL play data
            price: Price data
            price_columns: Price columns to include
            time_diff: Time difference between play and price

        Returns:
            Combined record dictionary
        """
        record = play.to_dict()

        # Add price data
        for col in price_columns:
            if col in price.index:
                record[f'price_{col}'] = price[col]

        # Add alignment metadata
        record['price_timestamp'] = price['timestamp']
        record['time_diff'] = time_diff.total_seconds()

        return record

    def _calculate_alignment_stats(
        self,
        aligned_data: pd.DataFrame,
        nfl_data: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate statistics about the alignment process.

        Args:
            aligned_data: Aligned DataFrame
            nfl_data: Original NFL data
            price_data: Original price data

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_nfl_plays': len(nfl_data),
            'total_price_points': len(price_data),
            'aligned_records': len(aligned_data),
            'alignment_rate': len(aligned_data) / len(nfl_data) if len(nfl_data) > 0 else 0.0,
            'unmatched_plays': len(nfl_data) - len(aligned_data),
            'time_range_nfl': {
                'start': nfl_data['timestamp'].min(),
                'end': nfl_data['timestamp'].max()
            } if len(nfl_data) > 0 else None,
            'time_range_price': {
                'start': price_data['timestamp'].min(),
                'end': price_data['timestamp'].max()
            } if len(price_data) > 0 else None,
        }

        if len(aligned_data) > 0 and 'time_diff' in aligned_data.columns:
            stats['time_diff_stats'] = {
                'mean': aligned_data['time_diff'].mean(),
                'median': aligned_data['time_diff'].median(),
                'std': aligned_data['time_diff'].std(),
                'max': aligned_data['time_diff'].max(),
                'min': aligned_data['time_diff'].min()
            }

        return stats

    def create_feature_matrix(
        self,
        aligned_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        lookback_window: int = 5
    ) -> pd.DataFrame:
        """Create a feature matrix for machine learning.

        Args:
            aligned_data: Aligned DataFrame
            feature_columns: Specific columns to use as features
            lookback_window: Number of previous records to include as features

        Returns:
            Feature matrix DataFrame
        """
        try:
            if feature_columns is None:
                # Default feature columns
                feature_columns = [col for col in aligned_data.columns
                                 if col.startswith('price_') or col in [
                                     'momentum_score', 'down', 'distance', 'field_position',
                                     'score_differential', 'time_remaining', 'quarter'
                                 ]]

            feature_matrix = []

            for i in range(lookback_window, len(aligned_data)):
                features = {}

                # Current record features
                current_record = aligned_data.iloc[i]
                for col in feature_columns:
                    if col in current_record:
                        features[f'current_{col}'] = current_record[col]

                # Historical features (lookback window)
                for j in range(lookback_window):
                    historical_record = aligned_data.iloc[i - j - 1]
                    for col in feature_columns:
                        if col in historical_record:
                            features[f'lag_{j+1}_{col}'] = historical_record[col]

                # Add metadata
                features['timestamp'] = current_record['timestamp']
                features['play_id'] = current_record.get('play_id', i)

                feature_matrix.append(features)

            result_df = pd.DataFrame(feature_matrix)
            self.logger.info(f"Created feature matrix with {len(result_df)} records and {len(result_df.columns)} features")

            return result_df

        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {e}")
            raise

    def save_aligned_data(self, result: AlignmentResult, output_path: Union[str, Path]):
        """Save aligned data and statistics to files.

        Args:
            result: AlignmentResult object
            output_path: Base output path
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = output_path.stem

        # Save main aligned data
        aligned_path = output_dir / f"{base_name}_aligned.parquet"
        result.aligned_data.to_parquet(aligned_path, index=False)

        # Save unmatched data
        if not result.unmatched_plays.empty:
            unmatched_plays_path = output_dir / f"{base_name}_unmatched_plays.parquet"
            result.unmatched_plays.to_parquet(unmatched_plays_path, index=False)

        if not result.unmatched_prices.empty:
            unmatched_prices_path = output_dir / f"{base_name}_unmatched_prices.parquet"
            result.unmatched_prices.to_parquet(unmatched_prices_path, index=False)

        # Save alignment statistics
        stats_path = output_dir / f"{base_name}_alignment_stats.json"
        import json
        with open(stats_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_stats = self._make_json_serializable(result.alignment_stats)
            json.dump(serializable_stats, f, indent=2, default=str)

        self.logger.info(f"Saved aligned data to {output_dir}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj