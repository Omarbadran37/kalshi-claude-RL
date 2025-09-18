"""Validation and visualization tools for feature engineering."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .feature_pipeline import FeaturePipelineResult
from .momentum_detector import MomentumEvent
from ..config import get_config, get_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class FeatureValidator:
    """Validates feature engineering results and performs statistical tests."""

    def __init__(self, config=None):
        """Initialize feature validator.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.FeatureValidator")

    def validate_features(self, pipeline_result: FeaturePipelineResult) -> Dict[str, Any]:
        """Comprehensive validation of feature engineering results.

        Args:
            pipeline_result: Result from feature pipeline

        Returns:
            Validation report dictionary
        """
        try:
            self.logger.info("Starting comprehensive feature validation")
            
            df = pipeline_result.features_df
            feature_names = pipeline_result.feature_names
            target_columns = pipeline_result.target_columns
            
            validation_report = {
                'data_quality': self._validate_data_quality(df, feature_names),
                'feature_statistics': self._calculate_feature_statistics(df, feature_names),
                'correlation_analysis': self._analyze_correlations(df, feature_names, target_columns),
                'predictive_power': self._assess_predictive_power(df, feature_names, target_columns),
                'stability_tests': self._test_feature_stability(df, feature_names),
                'momentum_validation': self._validate_momentum_events(pipeline_result.momentum_events, df)
            }
            
            # Overall validation score
            validation_report['overall_score'] = self._calculate_validation_score(validation_report)
            
            self.logger.info("Feature validation completed")
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error in feature validation: {e}")
            raise

    def _validate_data_quality(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Validate data quality metrics."""
        quality_report = {}
        
        # Missing data analysis
        missing_counts = df[feature_names].isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        quality_report['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'features_with_missing': (missing_counts > 0).sum(),
            'max_missing_percentage': missing_percentages.max(),
            'features_high_missing': missing_percentages[missing_percentages > 20].to_dict()
        }
        
        # Infinite/NaN values
        inf_counts = {}
        for col in feature_names:
            if col in df.columns:
                inf_count = np.isinf(df[col]).sum()
                inf_counts[col] = inf_count
        
        quality_report['infinite_values'] = {
            'total_infinite': sum(inf_counts.values()),
            'features_with_infinite': sum(1 for count in inf_counts.values() if count > 0)
        }
        
        # Constant features
        constant_features = []
        for col in feature_names:
            if col in df.columns and df[col].nunique() <= 1:
                constant_features.append(col)
        
        quality_report['constant_features'] = {
            'count': len(constant_features),
            'features': constant_features
        }
        
        # Duplicate features
        duplicate_pairs = []
        for i, col1 in enumerate(feature_names):
            for col2 in feature_names[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    if df[col1].equals(df[col2]):
                        duplicate_pairs.append((col1, col2))
        
        quality_report['duplicate_features'] = {
            'count': len(duplicate_pairs),
            'pairs': duplicate_pairs[:10]  # Top 10 duplicate pairs
        }
        
        return quality_report

    def _calculate_feature_statistics(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive feature statistics."""
        stats_report = {}
        
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_features:
            return {'error': 'No numeric features found'}
        
        # Basic statistics
        feature_stats = df[numeric_features].describe()
        
        # Skewness and kurtosis
        skewness = df[numeric_features].skew()
        kurtosis = df[numeric_features].kurtosis()
        
        # Outlier detection (IQR method)
        outlier_counts = {}
        for col in numeric_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        stats_report = {
            'basic_statistics': feature_stats.to_dict(),
            'skewness': skewness.to_dict(),
            'kurtosis': kurtosis.to_dict(),
            'outlier_counts': outlier_counts,
            'highly_skewed_features': skewness[abs(skewness) > 2].to_dict(),
            'features_with_many_outliers': {k: v for k, v in outlier_counts.items() if v > len(df) * 0.05}
        }
        
        return stats_report

    def _analyze_correlations(self, df: pd.DataFrame, feature_names: List[str],
                            target_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature correlations."""
        correlation_report = {}
        
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_features) < 2:
            return {'error': 'Insufficient numeric features for correlation analysis'}
        
        # Feature-feature correlations
        feature_corr = df[numeric_features].corr()
        
        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_value = feature_corr.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'feature1': feature_corr.columns[i],
                        'feature2': feature_corr.columns[j],
                        'correlation': corr_value
                    })
        
        correlation_report['feature_correlations'] = {
            'correlation_matrix_shape': feature_corr.shape,
            'high_correlation_pairs': high_corr_pairs[:20],  # Top 20 pairs
            'max_correlation': feature_corr.abs().values[np.triu_indices_from(feature_corr.values, k=1)].max()
        }
        
        # Feature-target correlations
        if target_columns:
            target_correlations = {}
            for target in target_columns:
                if target in df.columns:
                    target_corrs = {}
                    for feature in numeric_features:
                        if not df[feature].isnull().all() and not df[target].isnull().all():
                            # Remove NaN values for correlation calculation
                            valid_mask = ~(df[feature].isnull() | df[target].isnull())
                            if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                                corr, p_value = pearsonr(df[feature][valid_mask], df[target][valid_mask])
                                target_corrs[feature] = {'correlation': corr, 'p_value': p_value}
                    
                    target_correlations[target] = target_corrs
            
            correlation_report['target_correlations'] = target_correlations
        
        return correlation_report

    def _assess_predictive_power(self, df: pd.DataFrame, feature_names: List[str],
                               target_columns: List[str]) -> Dict[str, Any]:
        """Assess predictive power of features."""
        if not target_columns:
            return {'error': 'No target columns specified'}
        
        predictive_report = {}
        
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        for target in target_columns:
            if target not in df.columns:
                continue
            
            target_report = {}
            
            # Prepare data
            valid_mask = ~(df[target].isnull())
            if valid_mask.sum() < 10:
                continue
            
            X = df[numeric_features][valid_mask].fillna(0)
            y = df[target][valid_mask]
            
            try:
                # F-test scores
                f_scores, f_pvalues = f_regression(X, y)
                f_results = pd.DataFrame({
                    'feature': numeric_features,
                    'f_score': f_scores,
                    'p_value': f_pvalues
                }).sort_values('f_score', ascending=False)
                
                target_report['f_test_results'] = {
                    'top_features': f_results.head(10).to_dict('records'),
                    'significant_features': len(f_results[f_results['p_value'] < 0.05])
                }
                
                # Mutual information
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_results = pd.DataFrame({
                    'feature': numeric_features,
                    'mutual_info': mi_scores
                }).sort_values('mutual_info', ascending=False)
                
                target_report['mutual_info_results'] = {
                    'top_features': mi_results.head(10).to_dict('records'),
                    'avg_mutual_info': mi_scores.mean()
                }
                
            except Exception as e:
                target_report['error'] = str(e)
            
            predictive_report[target] = target_report
        
        return predictive_report

    def _test_feature_stability(self, df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Test feature stability across time periods."""
        stability_report = {}
        
        if 'timestamp' not in df.columns:
            return {'error': 'No timestamp column for stability testing'}
        
        # Split data into time periods
        df_sorted = df.sort_values('timestamp')
        split_point = len(df_sorted) // 2
        
        first_half = df_sorted.iloc[:split_point]
        second_half = df_sorted.iloc[split_point:]
        
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        stability_metrics = {}
        
        for feature in numeric_features:
            if feature in first_half.columns and feature in second_half.columns:
                # Mean difference
                mean1 = first_half[feature].mean()
                mean2 = second_half[feature].mean()
                mean_diff = abs(mean2 - mean1) / (abs(mean1) + 1e-8)
                
                # Standard deviation difference
                std1 = first_half[feature].std()
                std2 = second_half[feature].std()
                std_diff = abs(std2 - std1) / (abs(std1) + 1e-8)
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        first_half[feature].dropna(),
                        second_half[feature].dropna()
                    )
                except:
                    ks_stat, ks_pvalue = np.nan, np.nan
                
                stability_metrics[feature] = {
                    'mean_relative_change': mean_diff,
                    'std_relative_change': std_diff,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'stable': mean_diff < 0.5 and std_diff < 0.5 and (np.isnan(ks_pvalue) or ks_pvalue > 0.05)
                }
        
        # Summary
        stable_features = [f for f, metrics in stability_metrics.items() if metrics['stable']]
        unstable_features = [f for f, metrics in stability_metrics.items() if not metrics['stable']]
        
        stability_report = {
            'total_features_tested': len(stability_metrics),
            'stable_features': len(stable_features),
            'unstable_features': len(unstable_features),
            'stability_percentage': len(stable_features) / len(stability_metrics) * 100 if stability_metrics else 0,
            'feature_stability_details': stability_metrics,
            'most_unstable_features': sorted(
                [(f, m['mean_relative_change'] + m['std_relative_change']) 
                 for f, m in stability_metrics.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
        
        return stability_report

    def _validate_momentum_events(self, momentum_events: List[MomentumEvent], 
                                df: pd.DataFrame) -> Dict[str, Any]:
        """Validate momentum events detection."""
        if not momentum_events:
            return {'error': 'No momentum events to validate'}
        
        momentum_report = {
            'total_events': len(momentum_events),
            'event_types': {},
            'direction_distribution': {},
            'strength_distribution': {},
            'temporal_distribution': {}
        }
        
        # Event type distribution
        for event in momentum_events:
            event_type = event.event_type.value
            momentum_report['event_types'][event_type] = momentum_report['event_types'].get(event_type, 0) + 1
        
        # Direction distribution
        for event in momentum_events:
            direction = event.direction.value
            momentum_report['direction_distribution'][direction] = momentum_report['direction_distribution'].get(direction, 0) + 1
        
        # Strength statistics
        strengths = [event.strength for event in momentum_events]
        momentum_report['strength_distribution'] = {
            'mean': np.mean(strengths),
            'median': np.median(strengths),
            'std': np.std(strengths),
            'min': np.min(strengths),
            'max': np.max(strengths)
        }
        
        # Temporal distribution (events per hour)
        if momentum_events:
            timestamps = [event.timestamp for event in momentum_events]
            df_events = pd.DataFrame({'timestamp': timestamps})
            df_events['hour'] = pd.to_datetime(df_events['timestamp']).dt.hour
            hourly_counts = df_events['hour'].value_counts().to_dict()
            momentum_report['temporal_distribution'] = hourly_counts
        
        return momentum_report

    def _calculate_validation_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall validation score (0-100)."""
        score = 100.0
        
        # Data quality penalties
        data_quality = validation_report.get('data_quality', {})
        missing_data = data_quality.get('missing_data', {})
        if missing_data.get('max_missing_percentage', 0) > 50:
            score -= 20
        elif missing_data.get('max_missing_percentage', 0) > 20:
            score -= 10
        
        if data_quality.get('constant_features', {}).get('count', 0) > 0:
            score -= 5
        
        # Feature statistics penalties
        feature_stats = validation_report.get('feature_statistics', {})
        if len(feature_stats.get('highly_skewed_features', {})) > 10:
            score -= 10
        
        # Correlation analysis bonus/penalty
        correlation = validation_report.get('correlation_analysis', {})
        high_corr_pairs = correlation.get('feature_correlations', {}).get('high_correlation_pairs', [])
        if len(high_corr_pairs) > 20:
            score -= 15  # Too many highly correlated features
        
        # Predictive power bonus
        predictive = validation_report.get('predictive_power', {})
        if predictive and not predictive.get('error'):
            # Bonus for having predictive features
            score += 5
        
        # Stability bonus/penalty
        stability = validation_report.get('stability_tests', {})
        stability_pct = stability.get('stability_percentage', 0)
        if stability_pct > 80:
            score += 10
        elif stability_pct < 50:
            score -= 15
        
        return max(0, min(100, score))


class FeatureVisualizer:
    """Creates visualizations for feature engineering results."""

    def __init__(self, config=None):
        """Initialize feature visualizer.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.FeatureVisualizer")
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_comprehensive_report(self, pipeline_result: FeaturePipelineResult,
                                  validation_report: Dict[str, Any],
                                  output_dir: Union[str, Path]) -> Dict[str, str]:
        """Create comprehensive visualization report.

        Args:
            pipeline_result: Result from feature pipeline
            validation_report: Validation report
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            self.logger.info("Creating comprehensive visualization report")
            
            # 1. Correlation heatmap
            plot_files['correlation_heatmap'] = self._plot_correlation_heatmap(
                pipeline_result, output_dir / 'correlation_heatmap.png'
            )
            
            # 2. Feature importance plot
            plot_files['feature_importance'] = self._plot_feature_importance(
                pipeline_result, validation_report, output_dir / 'feature_importance.png'
            )
            
            # 3. Time series plots
            plot_files['time_series'] = self._plot_time_series(
                pipeline_result, output_dir / 'time_series_analysis.png'
            )
            
            # 4. Momentum events timeline
            plot_files['momentum_timeline'] = self._plot_momentum_timeline(
                pipeline_result, output_dir / 'momentum_timeline.png'
            )
            
            # 5. Feature distributions
            plot_files['feature_distributions'] = self._plot_feature_distributions(
                pipeline_result, output_dir / 'feature_distributions.png'
            )
            
            # 6. Validation summary
            plot_files['validation_summary'] = self._plot_validation_summary(
                validation_report, output_dir / 'validation_summary.png'
            )
            
            # 7. Price vs features scatter plots
            plot_files['price_feature_scatter'] = self._plot_price_feature_scatter(
                pipeline_result, output_dir / 'price_feature_scatter.png'
            )
            
            self.logger.info(f"Created {len(plot_files)} visualization plots")
            return plot_files
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            raise

    def _plot_correlation_heatmap(self, pipeline_result: FeaturePipelineResult, 
                                filepath: Path) -> str:
        """Plot correlation heatmap of top features."""
        df = pipeline_result.features_df
        feature_names = pipeline_result.feature_names[:50]  # Top 50 features
        
        # Select numeric features
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_features) < 2:
            return str(filepath)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=False, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   linewidths=0.1,
                   cbar_kws={"shrink": .5})
        
        plt.title('Feature Correlation Heatmap (Top 50 Features)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_feature_importance(self, pipeline_result: FeaturePipelineResult,
                               validation_report: Dict[str, Any], filepath: Path) -> str:
        """Plot feature importance based on validation results."""
        # Extract feature importance from validation report
        predictive_power = validation_report.get('predictive_power', {})
        
        if not predictive_power or 'error' in predictive_power:
            return str(filepath)
        
        # Get F-test results for first target
        first_target = list(predictive_power.keys())[0]
        f_test_results = predictive_power[first_target].get('f_test_results', {})
        top_features = f_test_results.get('top_features', [])
        
        if not top_features:
            return str(filepath)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # F-scores plot
        features = [f['feature'] for f in top_features[:20]]
        f_scores = [f['f_score'] for f in top_features[:20]]
        
        bars1 = ax1.barh(range(len(features)), f_scores, color='skyblue', edgecolor='navy')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('F-Score', fontsize=12)
        ax1.set_title('Top 20 Features by F-Test Score', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + max(f_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=9)
        
        # Mutual information plot (if available)
        mi_results = predictive_power[first_target].get('mutual_info_results', {})
        mi_features = mi_results.get('top_features', [])
        
        if mi_features:
            features_mi = [f['feature'] for f in mi_features[:20]]
            mi_scores = [f['mutual_info'] for f in mi_features[:20]]
            
            bars2 = ax2.barh(range(len(features_mi)), mi_scores, color='lightcoral', edgecolor='darkred')
            ax2.set_yticks(range(len(features_mi)))
            ax2.set_yticklabels(features_mi, fontsize=10)
            ax2.set_xlabel('Mutual Information Score', fontsize=12)
            ax2.set_title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + max(mi_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_time_series(self, pipeline_result: FeaturePipelineResult, filepath: Path) -> str:
        """Plot time series of key features and targets."""
        df = pipeline_result.features_df
        
        if 'timestamp' not in df.columns:
            return str(filepath)
        
        # Select key features and targets for plotting
        plot_columns = []
        
        # Add price columns
        price_cols = [col for col in df.columns if 'close_price' in col.lower()]
        plot_columns.extend(price_cols[:2])
        
        # Add volume columns
        volume_cols = [col for col in df.columns if 'volume' in col.lower() and 'ratio' not in col.lower()]
        plot_columns.extend(volume_cols[:1])
        
        # Add momentum columns
        momentum_cols = [col for col in df.columns if 'momentum' in col.lower()]
        plot_columns.extend(momentum_cols[:3])
        
        # Add target columns
        plot_columns.extend(pipeline_result.target_columns[:2])
        
        # Remove duplicates and filter existing columns
        plot_columns = list(set([col for col in plot_columns if col in df.columns]))
        
        if not plot_columns:
            return str(filepath)
        
        # Create subplots
        n_plots = min(len(plot_columns), 6)  # Maximum 6 subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        for i, col in enumerate(plot_columns[:n_plots]):
            ax = axes[i]
            
            # Plot time series
            ax.plot(df_sorted['timestamp'], df_sorted[col], linewidth=1, alpha=0.8)
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add trend line
            if len(df_sorted) > 10:
                x_numeric = np.arange(len(df_sorted))
                y_values = df_sorted[col].dropna()
                if len(y_values) > 10:
                    z = np.polyfit(x_numeric[:len(y_values)], y_values, 1)
                    p = np.poly1d(z)
                    ax.plot(df_sorted['timestamp'][:len(y_values)], p(x_numeric[:len(y_values)]), 
                           "r--", alpha=0.7, linewidth=2, label='Trend')
                    ax.legend()
        
        plt.suptitle('Time Series Analysis of Key Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_momentum_timeline(self, pipeline_result: FeaturePipelineResult, filepath: Path) -> str:
        """Plot momentum events timeline."""
        momentum_events = pipeline_result.momentum_events
        df = pipeline_result.features_df
        
        if not momentum_events or 'timestamp' not in df.columns:
            return str(filepath)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Price chart with momentum events
        if 'close_price' in df.columns:
            df_sorted = df.sort_values('timestamp')
            ax1.plot(df_sorted['timestamp'], df_sorted['close_price'], 
                    linewidth=1, color='blue', alpha=0.7, label='Close Price')
            
            # Mark momentum events
            for event in momentum_events:
                color = 'green' if event.direction.value == 'bullish' else 'red'
                alpha = min(0.3 + event.strength * 0.7, 1.0)
                
                # Find corresponding price
                event_time = pd.to_datetime(event.timestamp)
                closest_idx = (df_sorted['timestamp'] - event_time).abs().idxmin()
                event_price = df_sorted.loc[closest_idx, 'close_price']
                
                ax1.scatter(event_time, event_price, 
                          color=color, alpha=alpha, s=50 + event.strength * 100,
                          edgecolors='black', linewidth=0.5)
            
            ax1.set_title('Price Chart with Momentum Events', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Close Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Momentum strength over time
        event_times = [pd.to_datetime(event.timestamp) for event in momentum_events]
        event_strengths = [event.strength * (1 if event.direction.value == 'bullish' else -1) 
                          for event in momentum_events]
        
        ax2.scatter(event_times, event_strengths, 
                   c=['green' if s > 0 else 'red' for s in event_strengths],
                   alpha=0.7, s=50)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Momentum Strength Timeline', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Momentum Strength')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_feature_distributions(self, pipeline_result: FeaturePipelineResult, filepath: Path) -> str:
        """Plot distributions of key features."""
        df = pipeline_result.features_df
        feature_names = pipeline_result.feature_names
        
        # Select top features for distribution plotting
        numeric_features = [col for col in feature_names[:20] 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_features:
            return str(filepath)
        
        # Create subplots
        n_features = min(len(numeric_features), 12)  # Maximum 12 distributions
        n_rows = (n_features + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(numeric_features[:n_features]):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot histogram
            data = df[feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
                ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = data.mean()
                std_val = data.std()
                ax.text(0.7, 0.9, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_features, n_rows * 3):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_validation_summary(self, validation_report: Dict[str, Any], filepath: Path) -> str:
        """Plot validation summary dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Data quality metrics
        data_quality = validation_report.get('data_quality', {})
        missing_data = data_quality.get('missing_data', {})
        
        quality_metrics = [
            'Missing Data %',
            'Infinite Values',
            'Constant Features',
            'Duplicate Features'
        ]
        quality_values = [
            missing_data.get('max_missing_percentage', 0),
            data_quality.get('infinite_values', {}).get('total_infinite', 0),
            data_quality.get('constant_features', {}).get('count', 0),
            data_quality.get('duplicate_features', {}).get('count', 0)
        ]
        
        bars1 = ax1.bar(quality_metrics, quality_values, color=['red', 'orange', 'yellow', 'lightblue'])
        ax1.set_title('Data Quality Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count/Percentage')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(quality_values) * 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 2. Feature stability
        stability = validation_report.get('stability_tests', {})
        stability_pct = stability.get('stability_percentage', 0)
        instability_pct = 100 - stability_pct
        
        ax2.pie([stability_pct, instability_pct], 
               labels=['Stable Features', 'Unstable Features'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Feature Stability', fontsize=12, fontweight='bold')
        
        # 3. Correlation analysis
        correlation = validation_report.get('correlation_analysis', {})
        feature_corr = correlation.get('feature_correlations', {})
        high_corr_count = len(feature_corr.get('high_correlation_pairs', []))
        max_corr = feature_corr.get('max_correlation', 0)
        
        ax3.bar(['High Correlation Pairs', 'Max Correlation'], 
               [high_corr_count, max_corr * 100], 
               color=['purple', 'pink'])
        ax3.set_title('Correlation Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count / Percentage')
        
        # 4. Overall validation score
        overall_score = validation_report.get('overall_score', 0)
        
        # Create gauge chart for overall score
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax4.plot(theta, r, 'k-', linewidth=3)
        ax4.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # Score indicator
        score_angle = np.pi * (1 - overall_score / 100)
        ax4.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=4)
        
        ax4.set_xlim(0, np.pi)
        ax4.set_ylim(0, 1.2)
        ax4.set_xticks([0, np.pi/2, np.pi])
        ax4.set_xticklabels(['100', '50', '0'])
        ax4.set_yticks([])
        ax4.set_title(f'Overall Validation Score: {overall_score:.1f}', 
                     fontsize=12, fontweight='bold')
        
        plt.suptitle('Feature Engineering Validation Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)

    def _plot_price_feature_scatter(self, pipeline_result: FeaturePipelineResult, filepath: Path) -> str:
        """Plot scatter plots of features vs price movements."""
        df = pipeline_result.features_df
        
        # Find price change columns
        price_change_cols = [col for col in df.columns if 'price_change' in col.lower()]
        if not price_change_cols:
            return str(filepath)
        
        target_col = price_change_cols[0]
        
        # Select top features
        feature_names = pipeline_result.feature_names[:12]
        numeric_features = [col for col in feature_names 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_features:
            return str(filepath)
        
        # Create subplots
        n_features = min(len(numeric_features), 12)
        n_rows = (n_features + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(numeric_features[:n_features]):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Create scatter plot
            valid_mask = ~(df[feature].isnull() | df[target_col].isnull())
            if valid_mask.sum() > 10:
                x_data = df[feature][valid_mask]
                y_data = df[target_col][valid_mask]
                
                ax.scatter(x_data, y_data, alpha=0.6, s=20)
                
                # Add correlation coefficient
                if len(x_data) > 2:
                    corr, p_value = pearsonr(x_data, y_data)
                    ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_value:.3f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel(target_col, fontsize=10)
                ax.set_title(f'{feature} vs {target_col}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * 3):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.suptitle('Feature vs Price Movement Scatter Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)