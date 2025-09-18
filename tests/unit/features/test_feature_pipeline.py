"""Tests for FeaturePipeline class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

from src.nfl_trading.features.feature_pipeline import (
    FeaturePipeline, FeatureConfig, FeaturePipelineResult
)


class TestFeaturePipeline:
    """Test cases for FeaturePipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FeatureConfig(
            enable_game_features=True,
            enable_technical_features=True,
            enable_momentum_features=True,
            lookback_window=5,
            feature_windows=[5, 10],
            scaling_method='standard',
            imputation_method='mean',
            handle_outliers=True,
            enable_feature_selection=False,
            min_data_points=10
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create FeaturePipeline instance."""
        return FeaturePipeline(config)

    @pytest.fixture
    def sample_nfl_data(self):
        """Create sample NFL data."""
        base_time = datetime.now()
        
        data = {
            'timestamp': [base_time + timedelta(minutes=i*2) for i in range(20)],
            'play_type': ['pass', 'run', 'touchdown', 'punt'] * 5,
            'possession_team': ['DAL', 'DAL', 'DAL', 'PHI'] * 5,
            'field_position': [25, 35, 85, 45] * 5,
            'score_home': [0, 0, 7, 7] * 5,
            'score_away': [0, 0, 0, 3] * 5,
            'quarter': [1, 1, 1, 2] * 5,
            'time_remaining': [3600 - i*120 for i in range(20)],
            'yards_gained': [8, 12, 25, -2] * 5,
            'down': [1, 2, 1, 4] * 5,
            'distance': [10, 7, 10, 8] * 5
        }
        
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        base_time = datetime.now()
        np.random.seed(42)
        
        prices = 50 + np.cumsum(np.random.randn(25) * 0.02)
        
        data = {
            'timestamp': [base_time + timedelta(minutes=i*2) for i in range(25)],
            'close_price': prices,
            'open_price': prices + np.random.randn(25) * 0.01,
            'high_price': prices + np.abs(np.random.randn(25) * 0.02),
            'low_price': prices - np.abs(np.random.randn(25) * 0.02),
            'volume': np.random.randint(1000, 5000, 25),
            'bid_price': prices - 0.01,
            'ask_price': prices + 0.01
        }
        
        return pd.DataFrame(data)

    def test_initialization(self, pipeline, config):
        """Test FeaturePipeline initialization."""
        assert pipeline.config == config
        assert not pipeline.is_fitted
        assert pipeline.scaler is None
        assert pipeline.imputer is None

    def test_default_config(self):
        """Test default configuration."""
        pipeline = FeaturePipeline()
        assert isinstance(pipeline.config, FeatureConfig)
        assert pipeline.config.enable_game_features is True
        assert pipeline.config.scaling_method == 'standard'

    def test_fit_transform_basic(self, pipeline, sample_nfl_data, sample_price_data):
        """Test basic fit_transform functionality."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Check result type and properties
        assert isinstance(result, FeaturePipelineResult)
        assert isinstance(result.features_df, pd.DataFrame)
        assert isinstance(result.feature_names, list)
        assert isinstance(result.target_columns, list)
        assert isinstance(result.pipeline_stats, dict)
        
        # Check pipeline state
        assert pipeline.is_fitted
        
        # Check features were created
        assert len(result.feature_names) > 0
        assert len(result.features_df) > 0

    def test_validate_inputs(self, pipeline):
        """Test input validation."""
        # Valid inputs
        nfl_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'possession_team': ['DAL']
        })
        
        price_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close_price': [50.0],
            'volume': [1000]
        })
        
        validated_nfl, validated_price = pipeline._validate_inputs(nfl_data, price_data)
        assert 'timestamp' in validated_nfl.columns
        assert 'timestamp' in validated_price.columns
        
        # Missing required columns
        invalid_nfl = pd.DataFrame({'play_type': ['pass']})
        with pytest.raises(ValueError, match="NFL data missing required columns"):
            pipeline._validate_inputs(invalid_nfl, price_data)
        
        invalid_price = pd.DataFrame({'volume': [1000]})
        with pytest.raises(ValueError, match="Price data missing required columns"):
            pipeline._validate_inputs(nfl_data, invalid_price)

    @patch('src.nfl_trading.features.feature_pipeline.DataAligner')
    def test_data_alignment(self, mock_aligner, pipeline, sample_nfl_data, sample_price_data):
        """Test data alignment in pipeline."""
        # Mock alignment result
        mock_alignment_result = Mock()
        mock_alignment_result.aligned_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'close_price': [50.0]
        })
        
        mock_aligner.return_value.align_data.return_value = mock_alignment_result
        
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Check that aligner was called
        mock_aligner.return_value.align_data.assert_called_once()

    def test_feature_extraction_integration(self, pipeline, sample_nfl_data, sample_price_data):
        """Test integration of different feature extractors."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Check for different types of features
        feature_names = result.feature_names
        
        # Should have various feature types (if extractors worked)
        game_features = [f for f in feature_names if any(kw in f.lower() for kw in ['drive', 'situation', 'momentum'])]
        tech_features = [f for f in feature_names if any(kw in f.lower() for kw in ['sma', 'ema', 'rsi', 'bb_'])]
        
        # At least some features should be extracted
        assert len(feature_names) > 5

    def test_time_window_features(self, pipeline, sample_nfl_data, sample_price_data):
        """Test time window feature creation."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Check for time window features
        rolling_features = [f for f in result.feature_names if 'rolling' in f]
        change_features = [f for f in result.feature_names if 'change_' in f]
        
        # Should have some time-based features
        assert len(rolling_features) > 0 or len(change_features) > 0

    def test_target_creation(self, pipeline, sample_nfl_data, sample_price_data):
        """Test default target variable creation."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Should create default targets
        assert len(result.target_columns) > 0
        
        # Check for expected target types
        price_targets = [t for t in result.target_columns if 'price_change' in t]
        direction_targets = [t for t in result.target_columns if 'direction' in t]
        
        assert len(price_targets) > 0 or len(direction_targets) > 0

    def test_preprocessing_pipeline(self, pipeline, sample_nfl_data, sample_price_data):
        """Test preprocessing steps."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Check that preprocessing artifacts were stored
        assert 'scaler' in result.preprocessing_artifacts
        assert 'imputer' in result.preprocessing_artifacts
        assert 'feature_columns' in result.preprocessing_artifacts
        
        # Check pipeline components were fitted
        assert pipeline.scaler is not None
        assert pipeline.imputer is not None

    def test_feature_selection(self, sample_nfl_data, sample_price_data):
        """Test feature selection functionality."""
        config = FeatureConfig(
            enable_feature_selection=True,
            max_features=10,
            selection_method='f_regression'
        )
        pipeline = FeaturePipeline(config)
        
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Feature selection should limit number of features
        if len(result.feature_names) > 10:
            # Should have been reduced (if there were enough initial features)
            assert pipeline.feature_selector is not None

    def test_transform_after_fit(self, pipeline, sample_nfl_data, sample_price_data):
        """Test transform functionality after fitting."""
        # First fit the pipeline
        fit_result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Then transform new data
        transform_result = pipeline.transform(sample_nfl_data, sample_price_data)
        
        # Should have same structure
        assert len(transform_result.feature_names) == len(fit_result.feature_names)
        assert transform_result.feature_names == fit_result.feature_names

    def test_transform_without_fit_error(self, pipeline, sample_nfl_data, sample_price_data):
        """Test error when transforming without fitting."""
        with pytest.raises(ValueError, match="Pipeline must be fitted before transform"):
            pipeline.transform(sample_nfl_data, sample_price_data)

    def test_insufficient_data_error(self, pipeline):
        """Test error with insufficient data."""
        # Create minimal data that won't align well
        minimal_nfl = pd.DataFrame({
            'timestamp': [datetime.now()],
            'play_type': ['pass'],
            'possession_team': ['DAL']
        })
        
        minimal_price = pd.DataFrame({
            'timestamp': [datetime.now() + timedelta(hours=1)],  # Different time
            'close_price': [50.0],
            'volume': [1000]
        })
        
        with pytest.raises(ValueError, match="Insufficient aligned data points"):
            pipeline.fit_transform(minimal_nfl, minimal_price)

    def test_save_and_load_pipeline(self, pipeline, sample_nfl_data, sample_price_data):
        """Test saving and loading pipeline."""
        # Fit pipeline
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'test_pipeline.pkl'
            
            # Save pipeline
            pipeline.save_pipeline(filepath)
            assert filepath.exists()
            
            # Create new pipeline and load
            new_pipeline = FeaturePipeline()
            new_pipeline.load_pipeline(filepath)
            
            # Check that it was loaded correctly
            assert new_pipeline.is_fitted
            assert new_pipeline.scaler is not None
            assert new_pipeline.config.scaling_method == pipeline.config.scaling_method

    def test_save_without_fit_error(self, pipeline):
        """Test error when saving unfitted pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'test_pipeline.pkl'
            
            with pytest.raises(ValueError, match="Pipeline must be fitted before saving"):
                pipeline.save_pipeline(filepath)

    def test_load_nonexistent_file_error(self, pipeline):
        """Test error when loading nonexistent file."""
        nonexistent_path = Path('/nonexistent/path/pipeline.pkl')
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_pipeline(nonexistent_path)

    def test_pipeline_statistics(self, pipeline, sample_nfl_data, sample_price_data):
        """Test pipeline statistics calculation."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        stats = result.pipeline_stats
        
        # Check required statistics
        assert 'input_records' in stats
        assert 'output_records' in stats
        assert 'total_features' in stats
        assert 'processing_config' in stats
        assert 'feature_breakdown' in stats
        
        # Check config reflection
        config = stats['processing_config']
        assert config['game_features_enabled'] == pipeline.config.enable_game_features
        assert config['scaling_method'] == pipeline.config.scaling_method

    def test_feature_summary(self, pipeline, sample_nfl_data, sample_price_data):
        """Test feature summary generation."""
        # Test before fitting
        summary_unfitted = pipeline.get_feature_summary()
        assert 'error' in summary_unfitted
        
        # Fit pipeline
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
        
        # Test after fitting
        summary = pipeline.get_feature_summary()
        
        assert 'total_features' in summary
        assert 'feature_categories' in summary
        assert 'top_features' in summary
        assert 'preprocessing' in summary
        
        # Check preprocessing info
        preprocessing = summary['preprocessing']
        assert preprocessing['scaling_method'] == pipeline.config.scaling_method
        assert preprocessing['imputation_method'] == pipeline.config.imputation_method

    def test_outlier_handling(self, sample_nfl_data, sample_price_data):
        """Test outlier handling functionality."""
        config = FeatureConfig(handle_outliers=True, outlier_threshold=2.0)
        pipeline = FeaturePipeline(config)
        
        # Add some extreme outliers to price data
        price_data_with_outliers = sample_price_data.copy()
        price_data_with_outliers.loc[0, 'close_price'] = 1000  # Extreme outlier
        price_data_with_outliers.loc[1, 'volume'] = 1000000   # Extreme outlier
        
        result = pipeline.fit_transform(sample_nfl_data, price_data_with_outliers)
        
        # Pipeline should handle outliers without errors
        assert len(result.features_df) > 0

    def test_missing_data_handling(self, pipeline, sample_nfl_data, sample_price_data):
        """Test missing data handling."""
        # Add missing values
        nfl_with_missing = sample_nfl_data.copy()
        nfl_with_missing.loc[5:8, 'yards_gained'] = np.nan
        
        price_with_missing = sample_price_data.copy()
        price_with_missing.loc[3:6, 'volume'] = np.nan
        
        result = pipeline.fit_transform(nfl_with_missing, price_with_missing)
        
        # Should handle missing data gracefully
        assert len(result.features_df) > 0

    def test_different_scaling_methods(self, sample_nfl_data, sample_price_data):
        """Test different scaling methods."""
        scaling_methods = ['standard', 'minmax', 'robust']
        
        for method in scaling_methods:
            config = FeatureConfig(scaling_method=method)
            pipeline = FeaturePipeline(config)
            
            result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
            
            # Should work with all scaling methods
            assert len(result.features_df) > 0
            assert pipeline.scaler is not None

    def test_different_imputation_methods(self, sample_nfl_data, sample_price_data):
        """Test different imputation methods."""
        imputation_methods = ['mean', 'median', 'knn']
        
        for method in imputation_methods:
            config = FeatureConfig(imputation_method=method)
            pipeline = FeaturePipeline(config)
            
            result = pipeline.fit_transform(sample_nfl_data, sample_price_data)
            
            # Should work with all imputation methods
            assert len(result.features_df) > 0
            assert pipeline.imputer is not None

    def test_feature_config_validation(self):
        """Test FeatureConfig validation and defaults."""
        # Test default config
        config = FeatureConfig()
        assert config.enable_game_features is True
        assert config.enable_technical_features is True
        assert config.scaling_method == 'standard'
        assert config.min_data_points == 50
        
        # Test custom config
        custom_config = FeatureConfig(
            enable_game_features=False,
            scaling_method='minmax',
            min_data_points=20
        )
        assert custom_config.enable_game_features is False
        assert custom_config.scaling_method == 'minmax'
        assert custom_config.min_data_points == 20

    def test_pipeline_with_team_focus(self, pipeline, sample_nfl_data, sample_price_data):
        """Test pipeline with team focus parameter."""
        result = pipeline.fit_transform(sample_nfl_data, sample_price_data, team_focus='DAL')
        
        # Should work with team focus
        assert len(result.features_df) > 0
        
        # Should have team-specific features if game features enabled
        if pipeline.config.enable_game_features:
            team_features = [f for f in result.feature_names if 'team_' in f]
            assert len(team_features) > 0

    def test_custom_target_columns(self, pipeline, sample_nfl_data, sample_price_data):
        """Test pipeline with custom target columns."""
        # Add custom target to price data
        price_data_with_target = sample_price_data.copy()
        price_data_with_target['custom_target'] = np.random.randn(len(price_data_with_target))
        
        custom_targets = ['custom_target']
        result = pipeline.fit_transform(
            sample_nfl_data, 
            price_data_with_target, 
            target_columns=custom_targets
        )
        
        # Should use custom targets
        assert result.target_columns == custom_targets

    def test_performance_with_disabled_features(self, sample_nfl_data, sample_price_data):
        """Test pipeline performance with different feature combinations disabled."""
        # Test with only technical features
        config_tech_only = FeatureConfig(
            enable_game_features=False,
            enable_technical_features=True,
            enable_momentum_features=False
        )
        pipeline_tech = FeaturePipeline(config_tech_only)
        result_tech = pipeline_tech.fit_transform(sample_nfl_data, sample_price_data)
        
        # Should still work
        assert len(result_tech.features_df) > 0
        
        # Test with no feature extraction (should still have basic aligned data)
        config_minimal = FeatureConfig(
            enable_game_features=False,
            enable_technical_features=False,
            enable_momentum_features=False
        )
        pipeline_minimal = FeaturePipeline(config_minimal)
        result_minimal = pipeline_minimal.fit_transform(sample_nfl_data, sample_price_data)
        
        # Should still work with minimal features
        assert len(result_minimal.features_df) > 0