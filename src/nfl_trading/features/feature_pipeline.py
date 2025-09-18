"""Feature pipeline for orchestrating NFL trading feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from .game_state_extractor import GameStateExtractor
from .technical_indicators import TechnicalIndicators
from .momentum_detector import MomentumDetector, MomentumEvent
from ..data.data_aligner import DataAligner
from ..config import get_config, get_logger


logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    # Feature extraction
    enable_game_features: bool = True
    enable_technical_features: bool = True
    enable_momentum_features: bool = True
    
    # Time windows
    lookback_window: int = 10
    feature_windows: List[int] = None
    
    # Preprocessing
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    imputation_method: str = 'mean'   # 'mean', 'median', 'knn'
    handle_outliers: bool = True
    outlier_threshold: float = 3.0
    
    # Feature selection
    enable_feature_selection: bool = False
    max_features: Optional[int] = None
    selection_method: str = 'f_regression'  # 'f_regression', 'mutual_info'
    
    # Validation
    min_data_points: int = 50
    max_missing_ratio: float = 0.3


@dataclass
class FeaturePipelineResult:
    """Result from feature pipeline processing."""
    features_df: pd.DataFrame
    feature_names: List[str]
    target_columns: List[str]
    momentum_events: List[MomentumEvent]
    pipeline_stats: Dict[str, Any]
    preprocessing_artifacts: Dict[str, Any]


class FeaturePipeline:
    """Orchestrates comprehensive feature engineering for NFL trading."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature pipeline.

        Args:
            config: Feature pipeline configuration
        """
        self.config = config or FeatureConfig()
        if self.config.feature_windows is None:
            self.config.feature_windows = [5, 10, 20]
        
        self.logger = get_logger(f"{__name__}.FeaturePipeline")
        
        # Initialize feature extractors
        self.game_extractor = GameStateExtractor()
        self.technical_extractor = TechnicalIndicators()
        self.momentum_detector = MomentumDetector()
        self.data_aligner = DataAligner()
        
        # Preprocessing components
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.preprocessing_artifacts = {}
        
        # Feature tracking
        self.feature_importance_mapping = {}
        self.is_fitted = False

    def fit_transform(self, nfl_data: pd.DataFrame, price_data: pd.DataFrame,
                     team_focus: Optional[str] = None,
                     target_columns: Optional[List[str]] = None) -> FeaturePipelineResult:
        """Fit the pipeline and transform data.

        Args:
            nfl_data: NFL play-by-play data
            price_data: Kalshi price data
            team_focus: Team to focus analysis on
            target_columns: Target columns for prediction

        Returns:
            FeaturePipelineResult with processed features
        """
        try:
            self.logger.info("Starting feature pipeline fit_transform")
            
            # Validate inputs
            nfl_data, price_data = self._validate_inputs(nfl_data, price_data)
            
            # Align data
            self.logger.info("Aligning NFL and price data")
            alignment_result = self.data_aligner.align_data(nfl_data, price_data)
            aligned_data = alignment_result.aligned_data
            
            if len(aligned_data) < self.config.min_data_points:
                raise ValueError(f"Insufficient aligned data points: {len(aligned_data)} < {self.config.min_data_points}")
            
            # Extract features
            features_df = aligned_data.copy()
            momentum_events = []
            
            # Game state features
            if self.config.enable_game_features:
                self.logger.info("Extracting game state features")
                game_features = self.game_extractor.extract_features(features_df, team_focus)
                features_df = self._merge_features(features_df, game_features, 'game')
            
            # Technical indicators
            if self.config.enable_technical_features:
                self.logger.info("Extracting technical indicator features")
                tech_features = self.technical_extractor.extract_features(features_df)
                features_df = self._merge_features(features_df, tech_features, 'technical')
            
            # Momentum detection
            if self.config.enable_momentum_features:
                self.logger.info("Detecting momentum features")
                momentum_features, momentum_events = self.momentum_detector.detect_momentum(
                    features_df, team_focus
                )
                features_df = momentum_features
            
            # Add time-window features
            features_df = self._add_time_window_features(features_df)
            
            # Create target variables if not specified
            if target_columns is None:
                target_columns = self._create_default_targets(features_df)
            
            # Separate features from targets and metadata
            feature_columns = self._identify_feature_columns(features_df, target_columns)
            
            # Preprocessing
            features_df = self._preprocess_features(features_df, feature_columns, fit=True)
            
            # Feature selection
            if self.config.enable_feature_selection:
                features_df, feature_columns = self._select_features(
                    features_df, feature_columns, target_columns
                )
            
            # Calculate pipeline statistics
            pipeline_stats = self._calculate_pipeline_stats(features_df, aligned_data, momentum_events)
            
            # Update feature importance mapping
            self._update_feature_mapping()
            
            self.is_fitted = True
            
            result = FeaturePipelineResult(
                features_df=features_df,
                feature_names=feature_columns,
                target_columns=target_columns,
                momentum_events=momentum_events,
                pipeline_stats=pipeline_stats,
                preprocessing_artifacts=self.preprocessing_artifacts
            )
            
            self.logger.info(f"Feature pipeline completed: {len(feature_columns)} features, {len(features_df)} samples")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in feature pipeline: {e}")
            raise

    def transform(self, nfl_data: pd.DataFrame, price_data: pd.DataFrame,
                 team_focus: Optional[str] = None) -> FeaturePipelineResult:
        """Transform new data using fitted pipeline.

        Args:
            nfl_data: NFL play-by-play data
            price_data: Kalshi price data
            team_focus: Team to focus analysis on

        Returns:
            FeaturePipelineResult with processed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        try:
            self.logger.info("Transforming new data with fitted pipeline")
            
            # Validate inputs
            nfl_data, price_data = self._validate_inputs(nfl_data, price_data)
            
            # Align data
            alignment_result = self.data_aligner.align_data(nfl_data, price_data)
            aligned_data = alignment_result.aligned_data
            
            # Extract features (same process as fit)
            features_df = aligned_data.copy()
            momentum_events = []
            
            if self.config.enable_game_features:
                game_features = self.game_extractor.extract_features(features_df, team_focus)
                features_df = self._merge_features(features_df, game_features, 'game')
            
            if self.config.enable_technical_features:
                tech_features = self.technical_extractor.extract_features(features_df)
                features_df = self._merge_features(features_df, tech_features, 'technical')
            
            if self.config.enable_momentum_features:
                momentum_features, momentum_events = self.momentum_detector.detect_momentum(
                    features_df, team_focus
                )
                features_df = momentum_features
            
            # Add time-window features
            features_df = self._add_time_window_features(features_df)
            
            # Get feature columns from artifacts
            feature_columns = self.preprocessing_artifacts.get('feature_columns', [])
            target_columns = self.preprocessing_artifacts.get('target_columns', [])
            
            # Ensure all expected columns exist
            for col in feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0  # Fill missing columns with default values
            
            # Preprocessing (transform only)
            features_df = self._preprocess_features(features_df, feature_columns, fit=False)
            
            # Feature selection (transform only)
            if self.config.enable_feature_selection and self.feature_selector is not None:
                feature_data = features_df[feature_columns].fillna(0)
                selected_features = self.feature_selector.transform(feature_data)
                selected_columns = self.preprocessing_artifacts.get('selected_feature_names', feature_columns)
                
                # Update features dataframe
                features_df = features_df.drop(columns=feature_columns)
                for i, col in enumerate(selected_columns):
                    features_df[col] = selected_features[:, i]
                
                feature_columns = selected_columns
            
            pipeline_stats = self._calculate_pipeline_stats(features_df, aligned_data, momentum_events)
            
            result = FeaturePipelineResult(
                features_df=features_df,
                feature_names=feature_columns,
                target_columns=target_columns,
                momentum_events=momentum_events,
                pipeline_stats=pipeline_stats,
                preprocessing_artifacts=self.preprocessing_artifacts
            )
            
            self.logger.info(f"Transform completed: {len(feature_columns)} features, {len(features_df)} samples")
            return result
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise

    def _validate_inputs(self, nfl_data: pd.DataFrame, 
                        price_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate input data formats."""
        # Check required columns for NFL data
        nfl_required = ['timestamp', 'play_type', 'possession_team']
        nfl_missing = [col for col in nfl_required if col not in nfl_data.columns]
        if nfl_missing:
            raise ValueError(f"NFL data missing required columns: {nfl_missing}")
        
        # Check required columns for price data
        price_required = ['timestamp', 'close_price', 'volume']
        price_missing = [col for col in price_required if col not in price_data.columns]
        if price_missing:
            raise ValueError(f"Price data missing required columns: {price_missing}")
        
        # Ensure timestamps are datetime
        nfl_data = nfl_data.copy()
        price_data = price_data.copy()
        nfl_data['timestamp'] = pd.to_datetime(nfl_data['timestamp'])
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        return nfl_data, price_data

    def _merge_features(self, base_df: pd.DataFrame, feature_df: pd.DataFrame, 
                       prefix: str) -> pd.DataFrame:
        """Merge feature dataframe with base dataframe."""
        # Identify feature columns (exclude timestamp and existing columns)
        feature_cols = [col for col in feature_df.columns 
                       if col not in base_df.columns and col != 'timestamp']
        
        if not feature_cols:
            return base_df
        
        # Merge on timestamp or index
        if 'timestamp' in feature_df.columns:
            merged_df = base_df.merge(
                feature_df[['timestamp'] + feature_cols],
                on='timestamp',
                how='left'
            )
        else:
            # Merge on index
            for col in feature_cols:
                if col in feature_df.columns:
                    base_df[col] = feature_df[col]
            merged_df = base_df
        
        self.logger.debug(f"Merged {len(feature_cols)} {prefix} features")
        return merged_df

    def _add_time_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-window based aggregated features."""
        # Numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp-related columns
        exclude_cols = ['timestamp', 'quarter', 'down', 'play_id']
        numeric_cols = [col for col in numeric_cols if not any(ex in col.lower() for ex in exclude_cols)]
        
        for window in self.config.feature_windows:
            for col in numeric_cols:
                if col in df.columns:
                    # Rolling statistics
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    
                    # Change from window ago
                    df[f'{col}_change_{window}'] = df[col] - df[col].shift(window)
                    df[f'{col}_pct_change_{window}'] = df[col].pct_change(periods=window)
        
        self.logger.debug(f"Added time-window features for {len(self.config.feature_windows)} windows")
        return df

    def _create_default_targets(self, df: pd.DataFrame) -> List[str]:
        """Create default target variables for prediction."""
        targets = []
        
        # Price-based targets
        if 'close_price' in df.columns:
            # Future price changes
            for horizon in [1, 3, 5]:
                target_col = f'future_price_change_{horizon}'
                df[target_col] = df['close_price'].shift(-horizon) / df['close_price'] - 1
                targets.append(target_col)
            
            # Price direction
            df['price_direction_1'] = (df['close_price'].shift(-1) > df['close_price']).astype(int)
            targets.append('price_direction_1')
        
        # Volume-based targets
        if 'volume' in df.columns:
            df['future_volume_spike'] = (df['volume'].shift(-1) > df['volume'].rolling(20).mean() * 2).astype(int)
            targets.append('future_volume_spike')
        
        return targets

    def _identify_feature_columns(self, df: pd.DataFrame, target_columns: List[str]) -> List[str]:
        """Identify columns to use as features."""
        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'play_id', 'game_id', 'description'] + target_columns
        
        # Include numeric columns and engineered features
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Include if numeric or engineered feature
                if (pd.api.types.is_numeric_dtype(df[col]) or 
                    any(keyword in col.lower() for keyword in 
                        ['momentum', 'sma', 'ema', 'rsi', 'macd', 'bb_', 'volatility', 'trend'])):
                    feature_cols.append(col)
        
        self.logger.info(f"Identified {len(feature_cols)} feature columns")
        return feature_cols

    def _preprocess_features(self, df: pd.DataFrame, feature_columns: List[str], 
                           fit: bool = True) -> pd.DataFrame:
        """Preprocess features with scaling, imputation, and outlier handling."""
        if not feature_columns:
            return df
        
        # Extract feature data
        feature_data = df[feature_columns].copy()
        
        # Handle infinite values
        feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Imputation
        if fit:
            if self.config.imputation_method == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
            else:
                strategy = self.config.imputation_method
                self.imputer = SimpleImputer(strategy=strategy)
            
            imputed_data = self.imputer.fit_transform(feature_data)
        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted")
            imputed_data = self.imputer.transform(feature_data)
        
        # Convert back to DataFrame
        feature_data = pd.DataFrame(imputed_data, columns=feature_columns, index=feature_data.index)
        
        # Outlier handling
        if self.config.handle_outliers:
            feature_data = self._handle_outliers(feature_data, fit)
        
        # Scaling
        if fit:
            if self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            scaled_data = self.scaler.fit_transform(feature_data)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            scaled_data = self.scaler.transform(feature_data)
        
        # Update DataFrame
        for i, col in enumerate(feature_columns):
            df[col] = scaled_data[:, i]
        
        # Store preprocessing artifacts
        if fit:
            self.preprocessing_artifacts.update({
                'feature_columns': feature_columns,
                'imputer': self.imputer,
                'scaler': self.scaler
            })
        
        return df

    def _handle_outliers(self, feature_data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers using statistical methods."""
        if fit:
            # Calculate outlier bounds
            outlier_bounds = {}
            for col in feature_data.columns:
                q1 = feature_data[col].quantile(0.25)
                q3 = feature_data[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.config.outlier_threshold * iqr
                upper_bound = q3 + self.config.outlier_threshold * iqr
                
                outlier_bounds[col] = (lower_bound, upper_bound)
            
            self.preprocessing_artifacts['outlier_bounds'] = outlier_bounds
        else:
            outlier_bounds = self.preprocessing_artifacts.get('outlier_bounds', {})
        
        # Clip outliers
        for col, (lower, upper) in outlier_bounds.items():
            if col in feature_data.columns:
                feature_data[col] = feature_data[col].clip(lower, upper)
        
        return feature_data

    def _select_features(self, df: pd.DataFrame, feature_columns: List[str],
                        target_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using statistical methods."""
        if not target_columns or not feature_columns:
            return df, feature_columns
        
        # Use first target for feature selection
        target_col = target_columns[0]
        if target_col not in df.columns:
            return df, feature_columns
        
        # Prepare data
        X = df[feature_columns].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove samples with missing targets
        mask = ~pd.isna(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:  # Not enough data for feature selection
            return df, feature_columns
        
        # Feature selection
        k = min(self.config.max_features or len(feature_columns), len(feature_columns))
        
        if self.config.selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        
        try:
            selector.fit(X, y)
            selected_features = selector.transform(X)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_feature_names = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
            
            # Update dataframe
            df = df.drop(columns=feature_columns)
            for i, col in enumerate(selected_feature_names):
                df[col] = selected_features[:, i]
            
            # Store artifacts
            self.feature_selector = selector
            self.preprocessing_artifacts['selected_feature_names'] = selected_feature_names
            
            self.logger.info(f"Selected {len(selected_feature_names)} features from {len(feature_columns)}")
            return df, selected_feature_names
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}, using all features")
            return df, feature_columns

    def _calculate_pipeline_stats(self, features_df: pd.DataFrame, aligned_data: pd.DataFrame,
                                momentum_events: List[MomentumEvent]) -> Dict[str, Any]:
        """Calculate statistics about the pipeline processing."""
        stats = {
            'input_records': len(aligned_data),
            'output_records': len(features_df),
            'total_features': len(features_df.columns),
            'momentum_events': len(momentum_events),
            'missing_data_ratio': features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)),
            'processing_config': {
                'game_features_enabled': self.config.enable_game_features,
                'technical_features_enabled': self.config.enable_technical_features,
                'momentum_features_enabled': self.config.enable_momentum_features,
                'scaling_method': self.config.scaling_method,
                'imputation_method': self.config.imputation_method,
                'feature_selection_enabled': self.config.enable_feature_selection
            }
        }
        
        # Feature type breakdown
        feature_types = {
            'game_features': 0,
            'technical_features': 0,
            'momentum_features': 0,
            'time_window_features': 0,
            'other_features': 0
        }
        
        for col in features_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['game', 'drive', 'situation', 'time_']):
                feature_types['game_features'] += 1
            elif any(keyword in col_lower for keyword in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'volatility']):
                feature_types['technical_features'] += 1
            elif 'momentum' in col_lower:
                feature_types['momentum_features'] += 1
            elif any(keyword in col_lower for keyword in ['rolling', 'change_', 'pct_change']):
                feature_types['time_window_features'] += 1
            else:
                feature_types['other_features'] += 1
        
        stats['feature_breakdown'] = feature_types
        
        return stats

    def _update_feature_mapping(self):
        """Update feature importance mapping from all extractors."""
        self.feature_importance_mapping = {}
        
        # Game state features
        if self.config.enable_game_features:
            self.feature_importance_mapping.update(
                self.game_extractor.get_feature_importance_mapping()
            )
        
        # Technical features
        if self.config.enable_technical_features:
            self.feature_importance_mapping.update(
                self.technical_extractor.get_feature_importance_mapping()
            )
        
        # Momentum features
        if self.config.enable_momentum_features:
            self.feature_importance_mapping.update(
                self.momentum_detector.get_feature_importance_mapping()
            )

    def save_pipeline(self, filepath: Union[str, Path]):
        """Save fitted pipeline to disk."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        pipeline_data = {
            'config': self.config,
            'preprocessing_artifacts': self.preprocessing_artifacts,
            'feature_importance_mapping': self.feature_importance_mapping,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: Union[str, Path]):
        """Load fitted pipeline from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        
        self.config = pipeline_data['config']
        self.preprocessing_artifacts = pipeline_data['preprocessing_artifacts']
        self.feature_importance_mapping = pipeline_data['feature_importance_mapping']
        self.scaler = pipeline_data['scaler']
        self.imputer = pipeline_data['imputer']
        self.feature_selector = pipeline_data['feature_selector']
        self.is_fitted = pipeline_data['is_fitted']
        
        self.logger.info(f"Pipeline loaded from {filepath}")

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of extracted features."""
        if not self.is_fitted:
            return {"error": "Pipeline not fitted"}
        
        feature_columns = self.preprocessing_artifacts.get('feature_columns', [])
        
        summary = {
            'total_features': len(feature_columns),
            'feature_categories': {},
            'top_features': feature_columns[:20],  # Top 20 features
            'preprocessing': {
                'scaling_method': self.config.scaling_method,
                'imputation_method': self.config.imputation_method,
                'outlier_handling': self.config.handle_outliers,
                'feature_selection': self.config.enable_feature_selection
            }
        }
        
        # Categorize features
        categories = ['game', 'technical', 'momentum', 'rolling', 'change']
        for category in categories:
            summary['feature_categories'][category] = [
                col for col in feature_columns if category in col.lower()
            ]
        
        return summary