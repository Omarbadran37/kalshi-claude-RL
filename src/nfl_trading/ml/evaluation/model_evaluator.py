"""
Model Evaluator

Comprehensive evaluation metrics and analysis for NFL play analysis models.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F

from ..inference.play_predictor import PlayPredictor, PredictionResult

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    
    Features:
    - Classification and regression metrics
    - Calibration analysis
    - Trading performance evaluation
    - Detailed visualizations
    - Statistical significance tests
    """
    
    def __init__(self, predictor: PlayPredictor, output_dir: str = "evaluation_results"):
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.evaluation_results = {}
        self.predictions_cache = {}
        
        logger.info(f"ModelEvaluator initialized, output dir: {self.output_dir}")
    
    def comprehensive_evaluation(self,
                                test_data: List[Tuple[str, Dict, int, float]],
                                save_plots: bool = True,
                                include_trading_analysis: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation
        
        Args:
            test_data: List of (description, numerical_features, outcome_label, price_target) tuples
            save_plots: Whether to save visualization plots
            include_trading_analysis: Whether to include trading performance analysis
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        
        logger.info(f"Running comprehensive evaluation on {len(test_data)} samples")
        
        # Generate predictions
        predictions = self._generate_predictions(test_data)
        
        # Extract true values and predictions
        true_outcomes, true_prices, pred_outcomes, pred_prices, confidences = self._extract_predictions(
            test_data, predictions
        )
        
        # Classification evaluation
        classification_metrics = self._evaluate_classification(
            true_outcomes, pred_outcomes, predictions
        )
        
        # Regression evaluation
        regression_metrics = self._evaluate_regression(
            true_prices, pred_prices, predictions
        )
        
        # Calibration analysis
        calibration_metrics = self._evaluate_calibration(
            true_outcomes, predictions
        )
        
        # Trading performance evaluation
        trading_metrics = {}
        if include_trading_analysis:
            trading_metrics = self._evaluate_trading_performance(
                test_data, predictions
            )
        
        # Combine all results
        evaluation_results = {
            'classification_metrics': classification_metrics,
            'regression_metrics': regression_metrics,
            'calibration_metrics': calibration_metrics,
            'trading_metrics': trading_metrics,
            'data_statistics': self._calculate_data_statistics(test_data),
            'model_statistics': self.predictor.get_stats()
        }
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        # Generate plots
        if save_plots:
            self._create_evaluation_plots(
                test_data, predictions, evaluation_results
            )
        
        # Store for later access
        self.evaluation_results = evaluation_results
        
        logger.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def _generate_predictions(self, test_data: List[Tuple]) -> List[PredictionResult]:
        """Generate predictions for test data"""
        
        descriptions = [item[0] for item in test_data]
        numerical_features = [item[1] for item in test_data]
        
        # Batch prediction for efficiency
        predictions = self.predictor.predict_batch(
            descriptions, numerical_features, batch_size=32
        )
        
        return predictions
    
    def _extract_predictions(self, 
                           test_data: List[Tuple],
                           predictions: List[PredictionResult]) -> Tuple[List, List, List, List, List]:
        """Extract true values and predictions"""
        
        true_outcomes = [item[2] for item in test_data]
        true_prices = [item[3] for item in test_data]
        
        pred_outcomes = []
        pred_prices = []
        confidences = []
        
        for pred in predictions:
            # Get predicted outcome index
            outcome_labels = list(pred.outcome_probabilities.keys())
            pred_outcome_idx = outcome_labels.index(pred.predicted_outcome)
            pred_outcomes.append(pred_outcome_idx)
            
            pred_prices.append(pred.predicted_price_change)
            confidences.append((pred.outcome_confidence + pred.price_confidence) / 2)
        
        return true_outcomes, true_prices, pred_outcomes, pred_prices, confidences
    
    def _evaluate_classification(self, 
                                true_outcomes: List[int],
                                pred_outcomes: List[int],
                                predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Evaluate classification performance"""
        
        # Basic metrics
        accuracy = accuracy_score(true_outcomes, pred_outcomes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_outcomes, pred_outcomes, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_outcomes, pred_outcomes, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_outcomes, pred_outcomes)
        
        # Classification report
        class_report = classification_report(
            true_outcomes, pred_outcomes, output_dict=True
        )
        
        # ROC AUC (if binary or can be made binary)
        auc_scores = {}
        if len(set(true_outcomes)) == 2:
            # Binary classification
            probabilities = [pred.outcome_probabilities[pred.predicted_outcome] 
                           for pred in predictions]
            auc_scores['binary'] = roc_auc_score(true_outcomes, probabilities)
        else:
            # Multi-class: one-vs-rest AUC
            try:
                outcome_probs_matrix = []
                for pred in predictions:
                    probs = [pred.outcome_probabilities.get(f"outcome_{i}", 0.0) 
                           for i in range(len(pred.outcome_probabilities))]
                    outcome_probs_matrix.append(probs)
                
                outcome_probs_matrix = np.array(outcome_probs_matrix)
                auc_scores['macro'] = roc_auc_score(
                    true_outcomes, outcome_probs_matrix, multi_class='ovr', average='macro'
                )
            except Exception as e:
                logger.warning(f"Could not calculate multi-class AUC: {e}")
        
        # Confidence-based metrics
        confidence_scores = [pred.outcome_confidence for pred in predictions]
        
        # Top-k accuracy
        top_k_accuracies = {}
        for k in [1, 3, 5]:
            if k <= len(predictions[0].outcome_probabilities):
                top_k_acc = self._calculate_top_k_accuracy(
                    true_outcomes, predictions, k
                )
                top_k_accuracies[f'top_{k}'] = top_k_acc
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'auc_scores': auc_scores,
            'top_k_accuracies': top_k_accuracies,
            'avg_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores)
        }
    
    def _evaluate_regression(self,
                           true_prices: List[float],
                           pred_prices: List[float],
                           predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Evaluate regression performance"""
        
        # Basic metrics
        mse = mean_squared_error(true_prices, pred_prices)
        mae = mean_absolute_error(true_prices, pred_prices)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_prices, pred_prices)
        
        # MAPE (handling division by zero)
        try:
            mape = mean_absolute_percentage_error(true_prices, pred_prices)
        except:
            mape = np.mean(np.abs((np.array(true_prices) - np.array(pred_prices)) / 
                                (np.array(true_prices) + 1e-8))) * 100
        
        # Direction accuracy (sign prediction)
        true_directions = np.sign(true_prices)
        pred_directions = np.sign(pred_prices)
        direction_accuracy = accuracy_score(true_directions, pred_directions)
        
        # Confidence-weighted metrics
        price_confidences = [pred.price_confidence for pred in predictions]
        weighted_mae = np.average(
            np.abs(np.array(true_prices) - np.array(pred_prices)),
            weights=price_confidences
        )
        
        # Quantile analysis
        errors = np.abs(np.array(true_prices) - np.array(pred_prices))
        quantiles = {
            'q25': np.percentile(errors, 25),
            'q50': np.percentile(errors, 50),
            'q75': np.percentile(errors, 75),
            'q90': np.percentile(errors, 90),
            'q95': np.percentile(errors, 95)
        }
        
        # Prediction intervals (rough estimate)
        prediction_intervals = self._calculate_prediction_intervals(
            true_prices, pred_prices, price_confidences
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'weighted_mae': weighted_mae,
            'error_quantiles': quantiles,
            'prediction_intervals': prediction_intervals,
            'avg_price_confidence': np.mean(price_confidences),
            'price_confidence_std': np.std(price_confidences)
        }
    
    def _evaluate_calibration(self,
                            true_outcomes: List[int],
                            predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Evaluate model calibration"""
        
        # Extract confidence scores and binary correctness
        confidences = []
        correct_predictions = []
        
        for i, pred in enumerate(predictions):
            confidences.append(pred.outcome_confidence)
            
            # Check if prediction is correct
            outcome_labels = list(pred.outcome_probabilities.keys())
            pred_outcome_idx = outcome_labels.index(pred.predicted_outcome)
            correct_predictions.append(int(pred_outcome_idx == true_outcomes[i]))
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correct_predictions, confidences, n_bins=10
        )
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(correct_predictions, confidences, n_bins=10)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(correct_predictions, confidences, n_bins=10)
        
        # Brier score
        brier_score = np.mean((np.array(confidences) - np.array(correct_predictions)) ** 2)
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'brier_score': brier_score,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def _evaluate_trading_performance(self,
                                    test_data: List[Tuple],
                                    predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Evaluate trading performance"""
        
        # Simulate trading based on signals
        portfolio_value = 10000  # Starting capital
        positions = []
        trades = []
        
        for i, (description, numerical_features, true_outcome, true_price, pred) in enumerate(
            zip([item[0] for item in test_data],
                [item[1] for item in test_data], 
                [item[2] for item in test_data],
                [item[3] for item in test_data],
                predictions)
        ):
            
            # Simple trading logic based on signals
            if pred.trading_signal.value in ['buy', 'strong_buy']:
                # Buy signal
                position_size = min(100, portfolio_value * 0.1)  # Risk 10% per trade
                trade_return = true_price * position_size
                portfolio_value += trade_return
                
                trades.append({
                    'type': 'buy',
                    'signal': pred.trading_signal.value,
                    'signal_strength': pred.signal_strength,
                    'predicted_return': pred.predicted_price_change * position_size,
                    'actual_return': trade_return,
                    'confidence': pred.outcome_confidence
                })
                
            elif pred.trading_signal.value in ['sell', 'strong_sell']:
                # Sell signal (short)
                position_size = min(100, portfolio_value * 0.1)
                trade_return = -true_price * position_size  # Inverse for short
                portfolio_value += trade_return
                
                trades.append({
                    'type': 'sell',
                    'signal': pred.trading_signal.value,
                    'signal_strength': pred.signal_strength,
                    'predicted_return': -pred.predicted_price_change * position_size,
                    'actual_return': trade_return,
                    'confidence': pred.outcome_confidence
                })
        
        # Calculate trading metrics
        if trades:
            trade_returns = [trade['actual_return'] for trade in trades]
            predicted_returns = [trade['predicted_return'] for trade in trades]
            
            total_return = (portfolio_value - 10000) / 10000
            win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)
            avg_return_per_trade = np.mean(trade_returns)
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
            
            # Signal effectiveness
            signal_accuracy = self._calculate_signal_accuracy(trades)
            
            trading_metrics = {
                'total_return': total_return,
                'final_portfolio_value': portfolio_value,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_return_per_trade': avg_return_per_trade,
                'sharpe_ratio': sharpe_ratio,
                'signal_accuracy': signal_accuracy,
                'trade_history': trades
            }
        else:
            trading_metrics = {
                'total_return': 0.0,
                'final_portfolio_value': 10000,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'signal_accuracy': {},
                'trade_history': []
            }
        
        return trading_metrics
    
    def _calculate_top_k_accuracy(self, 
                                 true_outcomes: List[int],
                                 predictions: List[PredictionResult],
                                 k: int) -> float:
        """Calculate top-k accuracy"""
        
        correct = 0
        for i, pred in enumerate(predictions):
            # Get top k predictions
            outcome_probs = pred.outcome_probabilities
            sorted_outcomes = sorted(outcome_probs.items(), key=lambda x: x[1], reverse=True)
            top_k_outcomes = [outcome for outcome, _ in sorted_outcomes[:k]]
            
            # Check if true outcome is in top k
            true_outcome_label = f"outcome_{true_outcomes[i]}"
            if true_outcome_label in top_k_outcomes:
                correct += 1
        
        return correct / len(predictions)
    
    def _calculate_prediction_intervals(self,
                                      true_prices: List[float],
                                      pred_prices: List[float],
                                      confidences: List[float]) -> Dict[str, float]:
        """Calculate prediction interval coverage"""
        
        errors = np.abs(np.array(true_prices) - np.array(pred_prices))
        
        # Estimate intervals based on confidence
        intervals = {}
        for confidence_level in [0.8, 0.9, 0.95]:
            # Rough estimate: higher confidence -> tighter intervals
            interval_widths = np.array(confidences) * np.percentile(errors, 95)
            
            # Check coverage
            covered = sum(1 for i, error in enumerate(errors) 
                         if error <= interval_widths[i])
            coverage = covered / len(errors)
            
            intervals[f'coverage_{int(confidence_level*100)}'] = coverage
        
        return intervals
    
    def _calculate_ece(self, y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(y_true)[in_bin].mean()
                avg_confidence_in_bin = np.array(y_prob)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(y_true)[in_bin].mean()
                avg_confidence_in_bin = np.array(y_prob)[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_signal_accuracy(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy of trading signals"""
        
        signal_accuracy = {}
        
        for signal_type in ['buy', 'strong_buy', 'sell', 'strong_sell']:
            signal_trades = [t for t in trades if t['signal'] == signal_type]
            
            if signal_trades:
                correct_signals = sum(1 for t in signal_trades 
                                    if (t['type'] == 'buy' and t['actual_return'] > 0) or
                                       (t['type'] == 'sell' and t['actual_return'] > 0))
                signal_accuracy[signal_type] = correct_signals / len(signal_trades)
            else:
                signal_accuracy[signal_type] = 0.0
        
        return signal_accuracy
    
    def _calculate_data_statistics(self, test_data: List[Tuple]) -> Dict[str, Any]:
        """Calculate statistics about the test data"""
        
        outcomes = [item[2] for item in test_data]
        prices = [item[3] for item in test_data]
        
        return {
            'num_samples': len(test_data),
            'outcome_distribution': dict(zip(*np.unique(outcomes, return_counts=True))),
            'price_statistics': {
                'mean': np.mean(prices),
                'std': np.std(prices),
                'min': np.min(prices),
                'max': np.max(prices),
                'median': np.median(prices)
            }
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        
        results_path = self.output_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
    def _create_evaluation_plots(self,
                               test_data: List[Tuple],
                               predictions: List[PredictionResult],
                               results: Dict[str, Any]):
        """Create comprehensive evaluation plots"""
        
        logger.info("Creating evaluation plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
        cm = np.array(results['classification_metrics']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Classification Metrics Bar Plot
        ax2 = plt.subplot(3, 4, 2)
        class_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        class_values = [results['classification_metrics'][metric] for metric in class_metrics]
        ax2.bar(class_metrics, class_values)
        ax2.set_title('Classification Metrics')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(class_values):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. Price Prediction Scatter Plot
        ax3 = plt.subplot(3, 4, 3)
        true_prices = [item[3] for item in test_data]
        pred_prices = [pred.predicted_price_change for pred in predictions]
        ax3.scatter(true_prices, pred_prices, alpha=0.6)
        
        # Add perfect prediction line
        min_price = min(min(true_prices), min(pred_prices))
        max_price = max(max(true_prices), max(pred_prices))
        ax3.plot([min_price, max_price], [min_price, max_price], 'r--', label='Perfect Prediction')
        
        ax3.set_xlabel('True Price Change')
        ax3.set_ylabel('Predicted Price Change')
        ax3.set_title(f'Price Prediction (R² = {results["regression_metrics"]["r2_score"]:.3f})')
        ax3.legend()
        
        # 4. Residuals Plot
        ax4 = plt.subplot(3, 4, 4)
        residuals = np.array(true_prices) - np.array(pred_prices)
        ax4.scatter(pred_prices, residuals, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Price Change')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Plot')
        
        # 5. Calibration Plot
        ax5 = plt.subplot(3, 4, 5)
        calib_data = results['calibration_metrics']['calibration_curve']
        ax5.plot(calib_data['mean_predicted_value'], calib_data['fraction_of_positives'], 'o-', label='Model')
        ax5.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax5.set_xlabel('Mean Predicted Probability')
        ax5.set_ylabel('Fraction of Positives')
        ax5.set_title(f'Calibration Plot (ECE = {results["calibration_metrics"]["expected_calibration_error"]:.3f})')
        ax5.legend()
        
        # 6. Confidence Distribution
        ax6 = plt.subplot(3, 4, 6)
        confidences = [pred.outcome_confidence for pred in predictions]
        ax6.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Confidence Distribution')
        
        # 7. Trading Signals Distribution
        ax7 = plt.subplot(3, 4, 7)
        signals = [pred.trading_signal.value for pred in predictions]
        signal_counts = pd.Series(signals).value_counts()
        ax7.bar(signal_counts.index, signal_counts.values)
        ax7.set_title('Trading Signals Distribution')
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Error Distribution
        ax8 = plt.subplot(3, 4, 8)
        errors = np.abs(residuals)
        ax8.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Absolute Error')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Error Distribution')
        
        # 9. ROC Curve (if available)
        ax9 = plt.subplot(3, 4, 9)
        if 'binary' in results['classification_metrics']['auc_scores']:
            # Binary ROC curve
            true_outcomes = [item[2] for item in test_data]
            probabilities = [pred.outcome_confidence for pred in predictions]
            fpr, tpr, _ = roc_curve(true_outcomes, probabilities)
            auc = results['classification_metrics']['auc_scores']['binary']
            ax9.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            ax9.plot([0, 1], [0, 1], 'r--')
            ax9.set_xlabel('False Positive Rate')
            ax9.set_ylabel('True Positive Rate')
            ax9.set_title('ROC Curve')
            ax9.legend()
        else:
            ax9.text(0.5, 0.5, 'ROC Curve\nNot Available\n(Multi-class)', 
                    ha='center', va='center', transform=ax9.transAxes)
        
        # 10. Feature Importance (placeholder)
        ax10 = plt.subplot(3, 4, 10)
        # This would show feature importance if available
        ax10.text(0.5, 0.5, 'Feature Importance\n(See ExplainabilityAnalyzer)', 
                 ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Feature Importance')
        
        # 11. Portfolio Performance (if trading data available)
        ax11 = plt.subplot(3, 4, 11)
        if results['trading_metrics']['num_trades'] > 0:
            trade_returns = [t['actual_return'] for t in results['trading_metrics']['trade_history']]
            cumulative_returns = np.cumsum(trade_returns)
            ax11.plot(cumulative_returns)
            ax11.set_xlabel('Trade Number')
            ax11.set_ylabel('Cumulative Return')
            ax11.set_title(f'Portfolio Performance\n(Total Return: {results["trading_metrics"]["total_return"]:.2%})')
        else:
            ax11.text(0.5, 0.5, 'No Trading\nData Available', 
                     ha='center', va='center', transform=ax11.transAxes)
        
        # 12. Model Performance Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""Model Performance Summary
        
Classification:
• Accuracy: {results['classification_metrics']['accuracy']:.3f}
• F1 Score: {results['classification_metrics']['f1_score']:.3f}

Regression:
• R² Score: {results['regression_metrics']['r2_score']:.3f}
• MAE: {results['regression_metrics']['mae']:.4f}

Trading:
• Win Rate: {results['trading_metrics'].get('win_rate', 0):.2%}
• Sharpe: {results['trading_metrics'].get('sharpe_ratio', 0):.3f}

Calibration:
• ECE: {results['calibration_metrics']['expected_calibration_error']:.3f}
"""
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "comprehensive_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {plot_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        
        if not self.evaluation_results:
            return "No evaluation results available. Run comprehensive_evaluation first."
        
        results = self.evaluation_results
        
        report = f"""
NFL Play Analysis Model - Evaluation Report
==========================================

Data Statistics:
- Test samples: {results['data_statistics']['num_samples']}
- Outcome distribution: {results['data_statistics']['outcome_distribution']}

Classification Performance:
- Accuracy: {results['classification_metrics']['accuracy']:.4f}
- Precision: {results['classification_metrics']['precision']:.4f}
- Recall: {results['classification_metrics']['recall']:.4f}
- F1 Score: {results['classification_metrics']['f1_score']:.4f}

Regression Performance:
- R² Score: {results['regression_metrics']['r2_score']:.4f}
- MAE: {results['regression_metrics']['mae']:.6f}
- RMSE: {results['regression_metrics']['rmse']:.6f}
- Direction Accuracy: {results['regression_metrics']['direction_accuracy']:.4f}

Model Calibration:
- Expected Calibration Error: {results['calibration_metrics']['expected_calibration_error']:.4f}
- Brier Score: {results['calibration_metrics']['brier_score']:.4f}

Trading Performance:
- Total Return: {results['trading_metrics'].get('total_return', 0):.2%}
- Number of Trades: {results['trading_metrics'].get('num_trades', 0)}
- Win Rate: {results['trading_metrics'].get('win_rate', 0):.2%}
- Sharpe Ratio: {results['trading_metrics'].get('sharpe_ratio', 0):.4f}

Model Statistics:
- Device: {results['model_statistics']['device']}
- Total Predictions: {results['model_statistics']['prediction_count']}
- Avg Inference Time: {results['model_statistics']['avg_inference_time']:.4f}s
"""
        
        # Save report
        report_path = self.output_dir / "evaluation_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {report_path}")
        return report