"""Tests for validation and visualization components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt

from src.nfl_trading.features.validation_viz import FeatureValidator, FeatureVisualizer
from src.nfl_trading.features.feature_pipeline import FeaturePipelineResult
from src.nfl_trading.features.momentum_detector import MomentumEvent, MomentumType, MomentumDirection, EventType


class TestFeatureValidator:
    """Test cases for FeatureValidator."""

    @pytest.fixture
    def validator(self):
        """Create FeatureValidator instance."""
        return FeatureValidator()

    @pytest.fixture
    def sample_pipeline_result(self):
        """Create sample pipeline result for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create feature data
        feature_data = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create target data
        target_data = feature_data[:, 0] + np.random.randn(n_samples) * 0.1
        
        # Create dataframe
        df = pd.DataFrame(feature_data, columns=feature_names)
        df['target_1'] = target_data
        df['target_2'] = np.random.randn(n_samples)
        df['timestamp'] = [datetime.now() + timedelta(minutes=i) for i in range(n_samples)]
        
        # Create momentum events
        momentum_events = [
            MomentumEvent(
                timestamp=datetime.now() + timedelta(minutes=10),
                event_type=EventType.TOUCHDOWN,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BULLISH,
                strength=0.8,
                confidence=0.9,
                duration_estimate=timedelta(minutes=5),
                description="Test touchdown"
            ),
            MomentumEvent(
                timestamp=datetime.now() + timedelta(minutes=30),
                event_type=EventType.TURNOVER,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BEARISH,
                strength=0.7,
                confidence=0.8,
                duration_estimate=timedelta(minutes=3),
                description="Test turnover"
            )
        ]
        
        return FeaturePipelineResult(
            features_df=df,
            feature_names=feature_names,
            target_columns=['target_1', 'target_2'],
            momentum_events=momentum_events,
            pipeline_stats={'input_records': n_samples, 'output_records': n_samples},
            preprocessing_artifacts={}
        )

    def test_initialization(self, validator):
        """Test FeatureValidator initialization."""
        assert validator.config is not None
        assert validator.logger is not None

    def test_validate_features_basic(self, validator, sample_pipeline_result):
        """Test basic feature validation."""
        validation_report = validator.validate_features(sample_pipeline_result)
        
        # Check report structure
        assert isinstance(validation_report, dict)
        assert 'data_quality' in validation_report
        assert 'feature_statistics' in validation_report
        assert 'correlation_analysis' in validation_report
        assert 'predictive_power' in validation_report
        assert 'stability_tests' in validation_report
        assert 'momentum_validation' in validation_report
        assert 'overall_score' in validation_report

    def test_data_quality_validation(self, validator, sample_pipeline_result):
        """Test data quality validation."""
        # Add some quality issues
        df = sample_pipeline_result.features_df.copy()
        df.loc[0:5, 'feature_0'] = np.nan  # Missing data
        df.loc[10:15, 'feature_1'] = np.inf  # Infinite values
        df['constant_feature'] = 1  # Constant feature
        df['duplicate_feature'] = df['feature_0']  # Duplicate feature
        
        # Update result
        sample_pipeline_result.features_df = df
        sample_pipeline_result.feature_names.extend(['constant_feature', 'duplicate_feature'])
        
        quality_report = validator._validate_data_quality(df, sample_pipeline_result.feature_names)
        
        # Check missing data detection
        assert quality_report['missing_data']['total_missing'] > 0
        assert quality_report['missing_data']['features_with_missing'] > 0
        
        # Check infinite values detection
        assert quality_report['infinite_values']['total_infinite'] > 0
        
        # Check constant features detection
        assert quality_report['constant_features']['count'] > 0
        assert 'constant_feature' in quality_report['constant_features']['features']

    def test_feature_statistics(self, validator, sample_pipeline_result):
        """Test feature statistics calculation."""
        stats_report = validator._calculate_feature_statistics(
            sample_pipeline_result.features_df, 
            sample_pipeline_result.feature_names
        )
        
        # Check basic statistics
        assert 'basic_statistics' in stats_report
        assert 'skewness' in stats_report
        assert 'kurtosis' in stats_report
        assert 'outlier_counts' in stats_report
        
        # Check statistics structure
        basic_stats = stats_report['basic_statistics']
        assert 'mean' in basic_stats
        assert 'std' in basic_stats
        assert 'min' in basic_stats
        assert 'max' in basic_stats

    def test_correlation_analysis(self, validator, sample_pipeline_result):
        """Test correlation analysis."""
        correlation_report = validator._analyze_correlations(
            sample_pipeline_result.features_df,
            sample_pipeline_result.feature_names,
            sample_pipeline_result.target_columns
        )
        
        # Check feature correlations
        assert 'feature_correlations' in correlation_report
        feature_corr = correlation_report['feature_correlations']
        assert 'correlation_matrix_shape' in feature_corr
        assert 'high_correlation_pairs' in feature_corr
        
        # Check target correlations
        assert 'target_correlations' in correlation_report
        target_corr = correlation_report['target_correlations']
        assert len(target_corr) > 0

    def test_predictive_power_assessment(self, validator, sample_pipeline_result):
        """Test predictive power assessment."""
        predictive_report = validator._assess_predictive_power(
            sample_pipeline_result.features_df,
            sample_pipeline_result.feature_names,
            sample_pipeline_result.target_columns
        )
        
        # Should have results for each target
        assert len(predictive_report) > 0
        
        # Check first target report
        first_target = list(predictive_report.keys())[0]
        target_report = predictive_report[first_target]
        
        if 'error' not in target_report:
            assert 'f_test_results' in target_report
            assert 'mutual_info_results' in target_report

    def test_stability_testing(self, validator, sample_pipeline_result):
        """Test feature stability testing."""
        stability_report = validator._test_feature_stability(
            sample_pipeline_result.features_df,
            sample_pipeline_result.feature_names
        )
        
        # Check stability metrics
        assert 'total_features_tested' in stability_report
        assert 'stable_features' in stability_report
        assert 'unstable_features' in stability_report
        assert 'stability_percentage' in stability_report
        assert 'feature_stability_details' in stability_report

    def test_momentum_validation(self, validator, sample_pipeline_result):
        """Test momentum events validation."""
        momentum_report = validator._validate_momentum_events(
            sample_pipeline_result.momentum_events,
            sample_pipeline_result.features_df
        )
        
        # Check momentum validation
        assert 'total_events' in momentum_report
        assert 'event_types' in momentum_report
        assert 'direction_distribution' in momentum_report
        assert 'strength_distribution' in momentum_report
        
        # Check event counts
        assert momentum_report['total_events'] == len(sample_pipeline_result.momentum_events)

    def test_validation_score_calculation(self, validator):
        """Test overall validation score calculation."""
        # Create mock validation report
        validation_report = {
            'data_quality': {
                'missing_data': {'max_missing_percentage': 10},
                'constant_features': {'count': 1},
                'infinite_values': {'total_infinite': 0}
            },
            'feature_statistics': {
                'highly_skewed_features': {'feature_1': 3.5}
            },
            'correlation_analysis': {
                'feature_correlations': {'high_correlation_pairs': []}
            },
            'predictive_power': {
                'target_1': {'f_test_results': {'top_features': []}}
            },
            'stability_tests': {
                'stability_percentage': 85
            }
        }
        
        score = validator._calculate_validation_score(validation_report)
        
        # Score should be between 0 and 100
        assert 0 <= score <= 100

    def test_empty_data_handling(self, validator):
        """Test handling of empty or invalid data."""
        # Empty pipeline result
        empty_result = FeaturePipelineResult(
            features_df=pd.DataFrame(),
            feature_names=[],
            target_columns=[],
            momentum_events=[],
            pipeline_stats={},
            preprocessing_artifacts={}
        )
        
        # Should handle empty data gracefully
        validation_report = validator.validate_features(empty_result)
        assert isinstance(validation_report, dict)

    def test_edge_cases(self, validator, sample_pipeline_result):
        """Test edge cases in validation."""
        # Test with no target columns
        sample_pipeline_result.target_columns = []
        
        validation_report = validator.validate_features(sample_pipeline_result)
        
        # Should still work without targets
        assert 'data_quality' in validation_report
        assert 'feature_statistics' in validation_report


class TestFeatureVisualizer:
    """Test cases for FeatureVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create FeatureVisualizer instance."""
        return FeatureVisualizer()

    @pytest.fixture
    def sample_pipeline_result(self):
        """Create sample pipeline result for testing."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        # Create feature data
        feature_data = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create dataframe with various feature types
        df = pd.DataFrame(feature_data, columns=feature_names)
        df['close_price'] = 50 + np.cumsum(np.random.randn(n_samples) * 0.02)
        df['volume'] = np.random.randint(1000, 5000, n_samples)
        df['sma_5'] = df['close_price'].rolling(5).mean()
        df['rsi'] = np.random.uniform(20, 80, n_samples)
        df['momentum_score'] = np.random.randn(n_samples)
        df['target_1'] = df['feature_0'] + np.random.randn(n_samples) * 0.1
        df['timestamp'] = [datetime.now() + timedelta(minutes=i) for i in range(n_samples)]
        
        # Create momentum events
        momentum_events = [
            MomentumEvent(
                timestamp=datetime.now() + timedelta(minutes=10),
                event_type=EventType.TOUCHDOWN,
                momentum_type=MomentumType.GAME_MOMENTUM,
                direction=MomentumDirection.BULLISH,
                strength=0.8,
                confidence=0.9,
                duration_estimate=timedelta(minutes=5),
                description="Test touchdown"
            )
        ]
        
        return FeaturePipelineResult(
            features_df=df,
            feature_names=feature_names,
            target_columns=['target_1'],
            momentum_events=momentum_events,
            pipeline_stats={'input_records': n_samples, 'output_records': n_samples},
            preprocessing_artifacts={}
        )

    @pytest.fixture
    def sample_validation_report(self):
        """Create sample validation report."""
        return {
            'data_quality': {
                'missing_data': {'max_missing_percentage': 5, 'total_missing': 10},
                'infinite_values': {'total_infinite': 0},
                'constant_features': {'count': 1},
                'duplicate_features': {'count': 0}
            },
            'feature_statistics': {
                'basic_statistics': {'mean': {'feature_0': 0.1}, 'std': {'feature_0': 1.0}}
            },
            'correlation_analysis': {
                'feature_correlations': {'high_correlation_pairs': []}
            },
            'predictive_power': {
                'target_1': {
                    'f_test_results': {
                        'top_features': [
                            {'feature': 'feature_0', 'f_score': 10.5, 'p_value': 0.001},
                            {'feature': 'feature_1', 'f_score': 8.2, 'p_value': 0.005}
                        ]
                    },
                    'mutual_info_results': {
                        'top_features': [
                            {'feature': 'feature_0', 'mutual_info': 0.15},
                            {'feature': 'feature_1', 'mutual_info': 0.12}
                        ]
                    }
                }
            },
            'stability_tests': {
                'stability_percentage': 85,
                'stable_features': 8,
                'unstable_features': 2
            },
            'momentum_validation': {
                'total_events': 5,
                'event_types': {'touchdown': 2, 'turnover': 3}
            },
            'overall_score': 78.5
        }

    def test_initialization(self, visualizer):
        """Test FeatureVisualizer initialization."""
        assert visualizer.config is not None
        assert visualizer.logger is not None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_comprehensive_report_creation(self, mock_close, mock_savefig, 
                                         visualizer, sample_pipeline_result, sample_validation_report):
        """Test comprehensive visualization report creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            plot_files = visualizer.create_comprehensive_report(
                sample_pipeline_result, sample_validation_report, output_dir
            )
            
            # Check that plot files were created
            assert isinstance(plot_files, dict)
            assert len(plot_files) > 0
            
            # Check expected plot types
            expected_plots = [
                'correlation_heatmap',
                'feature_importance',
                'time_series',
                'momentum_timeline',
                'feature_distributions',
                'validation_summary'
            ]
            
            for plot_type in expected_plots:
                if plot_type in plot_files:
                    assert isinstance(plot_files[plot_type], str)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_correlation_heatmap(self, mock_close, mock_savefig, visualizer, sample_pipeline_result):
        """Test correlation heatmap creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'correlation_heatmap.png'
            
            result_path = visualizer._plot_correlation_heatmap(sample_pipeline_result, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)
            
            # Should call savefig
            mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_feature_importance_plot(self, mock_close, mock_savefig, 
                                   visualizer, sample_pipeline_result, sample_validation_report):
        """Test feature importance plot creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'feature_importance.png'
            
            result_path = visualizer._plot_feature_importance(
                sample_pipeline_result, sample_validation_report, filepath
            )
            
            # Should return filepath
            assert result_path == str(filepath)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_time_series_plot(self, mock_close, mock_savefig, visualizer, sample_pipeline_result):
        """Test time series plot creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'time_series.png'
            
            result_path = visualizer._plot_time_series(sample_pipeline_result, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_momentum_timeline_plot(self, mock_close, mock_savefig, visualizer, sample_pipeline_result):
        """Test momentum timeline plot creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'momentum_timeline.png'
            
            result_path = visualizer._plot_momentum_timeline(sample_pipeline_result, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_feature_distributions_plot(self, mock_close, mock_savefig, visualizer, sample_pipeline_result):
        """Test feature distributions plot creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'feature_distributions.png'
            
            result_path = visualizer._plot_feature_distributions(sample_pipeline_result, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_validation_summary_plot(self, mock_close, mock_savefig, 
                                   visualizer, sample_validation_report):
        """Test validation summary plot creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'validation_summary.png'
            
            result_path = visualizer._plot_validation_summary(sample_validation_report, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_price_feature_scatter_plot(self, mock_close, mock_savefig, visualizer, sample_pipeline_result):
        """Test price vs feature scatter plot creation."""
        # Add price change column
        df = sample_pipeline_result.features_df
        df['future_price_change_1'] = df['close_price'].pct_change().shift(-1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'price_feature_scatter.png'
            
            result_path = visualizer._plot_price_feature_scatter(sample_pipeline_result, filepath)
            
            # Should return filepath
            assert result_path == str(filepath)

    def test_empty_data_handling(self, visualizer):
        """Test handling of empty data in visualizations."""
        empty_result = FeaturePipelineResult(
            features_df=pd.DataFrame(),
            feature_names=[],
            target_columns=[],
            momentum_events=[],
            pipeline_stats={},
            preprocessing_artifacts={}
        )
        
        empty_validation = {}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Should handle empty data gracefully
            plot_files = visualizer.create_comprehensive_report(
                empty_result, empty_validation, output_dir
            )
            
            assert isinstance(plot_files, dict)

    def test_missing_columns_handling(self, visualizer, sample_pipeline_result):
        """Test handling of missing expected columns."""
        # Remove expected columns
        df = sample_pipeline_result.features_df
        df = df.drop(columns=['close_price', 'volume'], errors='ignore')
        sample_pipeline_result.features_df = df
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Should handle missing columns gracefully
            plot_files = visualizer.create_comprehensive_report(
                sample_pipeline_result, {}, output_dir
            )
            
            assert isinstance(plot_files, dict)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_no_numeric_features_handling(self, mock_close, mock_savefig, visualizer):
        """Test handling when no numeric features are available."""
        # Create result with only non-numeric features
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'text_feature': ['text'],
            'category_feature': ['A']
        })
        
        result = FeaturePipelineResult(
            features_df=df,
            feature_names=['text_feature', 'category_feature'],
            target_columns=[],
            momentum_events=[],
            pipeline_stats={},
            preprocessing_artifacts={}
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'correlation_heatmap.png'
            
            # Should handle gracefully
            result_path = visualizer._plot_correlation_heatmap(result, filepath)
            assert result_path == str(filepath)

    def test_plot_file_creation(self, visualizer, sample_pipeline_result, sample_validation_report):
        """Test that plot files are actually created in filesystem."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            
            # Mock matplotlib to avoid actual plotting but create files
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                with patch('matplotlib.pyplot.close'):
                    # Create a side effect that creates empty files
                    def create_file(filepath, **kwargs):
                        Path(filepath).touch()
                    
                    mock_savefig.side_effect = create_file
                    
                    plot_files = visualizer.create_comprehensive_report(
                        sample_pipeline_result, sample_validation_report, output_dir
                    )
                    
                    # Check that files were "created"
                    for plot_type, filepath in plot_files.items():
                        assert Path(filepath).exists()