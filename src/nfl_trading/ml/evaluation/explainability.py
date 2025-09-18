"""
Explainability Analyzer

Model interpretability and explainability tools for NFL play analysis models.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients, GradientShap, DeepLift, DeepLiftShap,
    Occlusion, LayerConductance, LayerIntegratedGradients
)

from ..inference.play_predictor import PlayPredictor, PredictionResult
from ..text_processor import PlayTextProcessor

logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis for NFL play models
    
    Features:
    - Attention weight visualization
    - Feature importance analysis
    - SHAP-like value attribution
    - Integrated gradients
    - Layer-wise relevance propagation
    - Text token importance
    """
    
    def __init__(self, 
                 predictor: PlayPredictor,
                 output_dir: str = "explainability_results"):
        
        self.predictor = predictor
        self.model = predictor.model
        self.text_processor = predictor.text_processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize attribution methods
        self._setup_attribution_methods()
        
        logger.info(f"ExplainabilityAnalyzer initialized, output dir: {self.output_dir}")
    
    def _setup_attribution_methods(self):
        """Setup various attribution methods"""
        
        # Wrapper for model forward function
        def forward_func(text_embeddings, numerical_features):
            attention_mask = torch.ones(text_embeddings.shape[0], 1).to(self.model.device)
            outputs = self.model(text_embeddings, numerical_features, attention_mask)
            return outputs['price_prediction'], outputs['outcome_logits']
        
        self.forward_func = forward_func
        
        # Attribution methods
        self.integrated_gradients = IntegratedGradients(forward_func)
        self.gradient_shap = GradientShap(forward_func)
        self.deep_lift = DeepLift(forward_func)
        
        logger.info("Attribution methods initialized")
    
    def analyze_play_explanation(self,
                                play_description: str,
                                numerical_features: Optional[Dict[str, float]] = None,
                                methods: List[str] = None,
                                save_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive explanation analysis for a single play
        
        Args:
            play_description: Play description text
            numerical_features: Numerical game features
            methods: Attribution methods to use
            save_plots: Whether to save visualization plots
        
        Returns:
            Dictionary with explanation results
        """
        
        if methods is None:
            methods = ['attention', 'integrated_gradients', 'feature_importance']
        
        logger.info(f"Analyzing explanation for play: {play_description[:50]}...")
        
        # Generate prediction first
        prediction = self.predictor.predict_play(
            play_description, 
            numerical_features, 
            return_feature_importance=True,
            return_attention=True
        )
        
        explanation_results = {
            'prediction': prediction.to_dict(),
            'explanations': {}
        }
        
        # Process the play
        processed_play = self.text_processor.process_play(play_description)
        
        if processed_play.embeddings is None:
            logger.warning("Could not process play for explanation")
            return explanation_results
        
        # Prepare inputs
        text_embedding = processed_play.embeddings.unsqueeze(0).to(self.model.device)
        numerical_tensor = self._prepare_numerical_features(processed_play, numerical_features)
        numerical_tensor = numerical_tensor.unsqueeze(0).to(self.model.device)
        
        # Attention analysis
        if 'attention' in methods:
            attention_explanation = self._analyze_attention_weights(
                text_embedding, numerical_tensor, processed_play
            )
            explanation_results['explanations']['attention'] = attention_explanation
        
        # Integrated gradients
        if 'integrated_gradients' in methods:
            ig_explanation = self._analyze_integrated_gradients(
                text_embedding, numerical_tensor, processed_play
            )
            explanation_results['explanations']['integrated_gradients'] = ig_explanation
        
        # Feature importance
        if 'feature_importance' in methods:
            feature_explanation = self._analyze_feature_importance(
                text_embedding, numerical_tensor, processed_play, numerical_features
            )
            explanation_results['explanations']['feature_importance'] = feature_explanation
        
        # Gradient SHAP
        if 'gradient_shap' in methods:
            shap_explanation = self._analyze_gradient_shap(
                text_embedding, numerical_tensor, processed_play
            )
            explanation_results['explanations']['gradient_shap'] = shap_explanation
        
        # Save visualizations
        if save_plots:
            self._create_explanation_plots(
                explanation_results, processed_play, play_description
            )
        
        return explanation_results
    
    def analyze_feature_importance_global(self,
                                        test_data: List[Tuple[str, Dict, int, float]],
                                        sample_size: int = 100) -> Dict[str, Any]:
        """
        Global feature importance analysis across multiple plays
        
        Args:
            test_data: Test data samples
            sample_size: Number of samples to analyze
        
        Returns:
            Global feature importance results
        """
        
        logger.info(f"Analyzing global feature importance on {sample_size} samples")
        
        # Sample data for analysis
        if len(test_data) > sample_size:
            import random
            sampled_data = random.sample(test_data, sample_size)
        else:
            sampled_data = test_data
        
        # Collect attributions
        text_attributions = []
        numerical_attributions = []
        feature_names = []
        
        for i, (description, num_features, outcome, price) in enumerate(sampled_data):
            if i % 20 == 0:
                logger.info(f"Processing sample {i+1}/{len(sampled_data)}")
            
            try:
                # Process play
                processed_play = self.text_processor.process_play(description)
                if processed_play.embeddings is None:
                    continue
                
                # Prepare inputs
                text_embedding = processed_play.embeddings.unsqueeze(0).to(self.model.device)
                numerical_tensor = self._prepare_numerical_features(processed_play, num_features)
                numerical_tensor = numerical_tensor.unsqueeze(0).to(self.model.device)
                
                # Get feature names on first iteration
                if not feature_names:
                    feature_names = self._get_feature_names(num_features)
                
                # Calculate attributions
                attributions = self._calculate_attributions(text_embedding, numerical_tensor)
                
                text_attributions.append(attributions['text'].cpu().numpy())
                numerical_attributions.append(attributions['numerical'].cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not text_attributions:
            logger.error("No valid attributions calculated")
            return {}
        
        # Aggregate results
        text_attributions = np.array(text_attributions)
        numerical_attributions = np.array(numerical_attributions)
        
        # Calculate global importance scores
        global_text_importance = np.mean(np.abs(text_attributions), axis=0)
        global_numerical_importance = np.mean(np.abs(numerical_attributions), axis=0)
        
        # Feature ranking
        if len(feature_names) == len(global_numerical_importance):
            feature_ranking = sorted(
                zip(feature_names, global_numerical_importance),
                key=lambda x: x[1], reverse=True
            )
        else:
            feature_ranking = []
        
        results = {
            'global_text_importance': global_text_importance.tolist(),
            'global_numerical_importance': global_numerical_importance.tolist(),
            'feature_names': feature_names,
            'feature_ranking': feature_ranking,
            'num_samples': len(text_attributions),
            'text_importance_std': np.std(text_attributions, axis=0).tolist(),
            'numerical_importance_std': np.std(numerical_attributions, axis=0).tolist()
        }
        
        # Save results
        self._save_global_importance_results(results)
        
        return results
    
    def _analyze_attention_weights(self,
                                 text_embedding: torch.Tensor,
                                 numerical_tensor: torch.Tensor,
                                 processed_play) -> Dict[str, Any]:
        """Analyze attention weights"""
        
        # Get attention weights from model
        attention_mask = torch.ones(1, 1).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                text_embedding, numerical_tensor, attention_mask, return_attention=True
            )
        
        attention_explanation = {'available': False}
        
        if 'attention_weights' in outputs and outputs['attention_weights']:
            # Extract attention weights
            attention_weights = outputs['attention_weights']
            
            # Average across heads and layers
            if isinstance(attention_weights, list):
                # Multiple layers
                avg_attention = torch.stack(attention_weights).mean(dim=0)
            else:
                avg_attention = attention_weights
            
            # Average across heads if multi-head
            if avg_attention.dim() > 3:
                avg_attention = avg_attention.mean(dim=1)
            
            attention_explanation = {
                'available': True,
                'attention_weights': avg_attention.cpu().numpy().tolist(),
                'attention_summary': {
                    'max_attention': float(avg_attention.max()),
                    'min_attention': float(avg_attention.min()),
                    'mean_attention': float(avg_attention.mean()),
                    'std_attention': float(avg_attention.std())
                }
            }
        
        return attention_explanation
    
    def _analyze_integrated_gradients(self,
                                    text_embedding: torch.Tensor,
                                    numerical_tensor: torch.Tensor,
                                    processed_play) -> Dict[str, Any]:
        """Analyze using integrated gradients"""
        
        try:
            # Baselines (typically zeros)
            text_baseline = torch.zeros_like(text_embedding)
            numerical_baseline = torch.zeros_like(numerical_tensor)
            
            # Calculate attributions for both outputs
            price_attributions = self.integrated_gradients.attribute(
                (text_embedding, numerical_tensor),
                baselines=(text_baseline, numerical_baseline),
                target=0,  # Price prediction (first output)
                n_steps=20  # Reduced for efficiency
            )
            
            outcome_attributions = self.integrated_gradients.attribute(
                (text_embedding, numerical_tensor),
                baselines=(text_baseline, numerical_baseline),
                target=1,  # Outcome prediction (second output)
                n_steps=20
            )
            
            # Combine attributions
            text_attr_combined = price_attributions[0] + outcome_attributions[0]
            numerical_attr_combined = price_attributions[1] + outcome_attributions[1]
            
            return {
                'text_attributions': text_attr_combined.cpu().numpy().tolist(),
                'numerical_attributions': numerical_attr_combined.cpu().numpy().tolist(),
                'price_text_attributions': price_attributions[0].cpu().numpy().tolist(),
                'price_numerical_attributions': price_attributions[1].cpu().numpy().tolist(),
                'outcome_text_attributions': outcome_attributions[0].cpu().numpy().tolist(),
                'outcome_numerical_attributions': outcome_attributions[1].cpu().numpy().tolist(),
                'attribution_summary': {
                    'text_importance': float(torch.norm(text_attr_combined)),
                    'numerical_importance': float(torch.norm(numerical_attr_combined))
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in integrated gradients analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self,
                                  text_embedding: torch.Tensor,
                                  numerical_tensor: torch.Tensor,
                                  processed_play,
                                  numerical_features: Optional[Dict]) -> Dict[str, Any]:
        """Analyze individual feature importance"""
        
        # Get feature names
        feature_names = self._get_feature_names(numerical_features)
        
        # Calculate feature importance using occlusion
        try:
            # Occlusion analysis for numerical features
            occlusion = Occlusion(self.forward_func)
            
            # Test each numerical feature
            feature_importances = {}
            
            for i, feature_name in enumerate(feature_names):
                # Create mask for this feature
                if i < numerical_tensor.shape[1]:
                    # Temporarily set feature to zero
                    modified_tensor = numerical_tensor.clone()
                    original_value = modified_tensor[0, i].item()
                    modified_tensor[0, i] = 0
                    
                    # Calculate difference in prediction
                    with torch.no_grad():
                        original_output = self.model(text_embedding, numerical_tensor, 
                                                   torch.ones(1, 1).to(self.model.device))
                        modified_output = self.model(text_embedding, modified_tensor,
                                                   torch.ones(1, 1).to(self.model.device))
                    
                    # Calculate importance as difference in predictions
                    price_diff = abs(original_output['price_prediction'].item() - 
                                   modified_output['price_prediction'].item())
                    outcome_diff = torch.norm(original_output['outcome_logits'] - 
                                            modified_output['outcome_logits']).item()
                    
                    feature_importances[feature_name] = {
                        'price_importance': price_diff,
                        'outcome_importance': outcome_diff,
                        'combined_importance': price_diff + outcome_diff,
                        'original_value': original_value
                    }
            
            # Sort by importance
            sorted_features = sorted(
                feature_importances.items(),
                key=lambda x: x[1]['combined_importance'],
                reverse=True
            )
            
            return {
                'feature_importances': feature_importances,
                'sorted_features': sorted_features,
                'top_features': sorted_features[:5]
            }
            
        except Exception as e:
            logger.warning(f"Error in feature importance analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_gradient_shap(self,
                             text_embedding: torch.Tensor,
                             numerical_tensor: torch.Tensor,
                             processed_play) -> Dict[str, Any]:
        """Analyze using Gradient SHAP"""
        
        try:
            # Create baseline distribution
            n_samples = 5  # Reduced for efficiency
            text_baselines = torch.randn(n_samples, *text_embedding.shape[1:]).to(self.model.device)
            numerical_baselines = torch.randn(n_samples, *numerical_tensor.shape[1:]).to(self.model.device)
            
            # Calculate SHAP values
            shap_values = self.gradient_shap.attribute(
                (text_embedding, numerical_tensor),
                baselines=(text_baselines, numerical_baselines),
                n_samples=n_samples,
                stdevs=0.1
            )
            
            return {
                'text_shap_values': shap_values[0].cpu().numpy().tolist(),
                'numerical_shap_values': shap_values[1].cpu().numpy().tolist(),
                'shap_summary': {
                    'text_total_attribution': float(torch.sum(shap_values[0])),
                    'numerical_total_attribution': float(torch.sum(shap_values[1]))
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in Gradient SHAP analysis: {e}")
            return {'error': str(e)}
    
    def _prepare_numerical_features(self,
                                  processed_play,
                                  additional_features: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Prepare numerical features (same as in PlayPredictor)"""
        
        # This should match the PlayPredictor implementation
        features = [
            len(processed_play.entities.players),
            len(processed_play.entities.actions),
            len(processed_play.entities.outcomes),
            processed_play.entities.yardage or 0,
            processed_play.entities.down or 0,
            processed_play.entities.distance or 0,
            processed_play.entities.score_change or 0,
            processed_play.confidence,
            len(processed_play.original_text),
            len(processed_play.cleaned_text.split()),
            len(processed_play.normalized_text.split())
        ]
        
        # Add additional features if provided
        if additional_features:
            feature_order = [
                'quarter', 'time_remaining', 'down', 'distance', 'field_position',
                'score_home', 'score_away', 'timeouts_home', 'timeouts_away'
            ]
            
            for feature_name in feature_order:
                features.append(additional_features.get(feature_name, 0.0))
        else:
            features.extend([0.0] * 9)
        
        # Action type one-hot encoding
        action_types = ['pass', 'rush', 'kick', 'turnover', 'penalty', 'score']
        for action_type in action_types:
            features.append(1.0 if action_type in processed_play.entities.actions else 0.0)
        
        # Outcome type one-hot encoding
        outcome_types = ['success', 'failure', 'neutral']
        for outcome_type in outcome_types:
            features.append(1.0 if outcome_type in processed_play.entities.outcomes else 0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_feature_names(self, numerical_features: Optional[Dict] = None) -> List[str]:
        """Get names of numerical features"""
        
        base_features = [
            'num_players', 'num_actions', 'num_outcomes', 'yardage', 'down', 
            'distance', 'score_change', 'confidence', 'text_length', 
            'cleaned_words', 'normalized_words'
        ]
        
        game_features = [
            'quarter', 'time_remaining', 'down', 'distance', 'field_position',
            'score_home', 'score_away', 'timeouts_home', 'timeouts_away'
        ]
        
        action_features = [f'action_{action}' for action in 
                          ['pass', 'rush', 'kick', 'turnover', 'penalty', 'score']]
        
        outcome_features = [f'outcome_{outcome}' for outcome in 
                           ['success', 'failure', 'neutral']]
        
        return base_features + game_features + action_features + outcome_features
    
    def _calculate_attributions(self, 
                              text_embedding: torch.Tensor,
                              numerical_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate attributions using integrated gradients"""
        
        text_baseline = torch.zeros_like(text_embedding)
        numerical_baseline = torch.zeros_like(numerical_tensor)
        
        # Calculate for price prediction
        attributions = self.integrated_gradients.attribute(
            (text_embedding, numerical_tensor),
            baselines=(text_baseline, numerical_baseline),
            target=0,
            n_steps=10
        )
        
        return {
            'text': attributions[0].squeeze(),
            'numerical': attributions[1].squeeze()
        }
    
    def _create_explanation_plots(self,
                                explanation_results: Dict[str, Any],
                                processed_play,
                                play_description: str):
        """Create visualization plots for explanations"""
        
        logger.info("Creating explanation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Explanation Analysis: {play_description[:50]}...', fontsize=14)
        
        # 1. Prediction Summary
        ax1 = axes[0, 0]
        prediction = explanation_results['prediction']
        
        # Show top predicted outcomes
        outcome_probs = prediction['outcome_probabilities']
        top_outcomes = sorted(outcome_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        outcomes, probs = zip(*top_outcomes)
        ax1.barh(range(len(outcomes)), probs)
        ax1.set_yticks(range(len(outcomes)))
        ax1.set_yticklabels(outcomes)
        ax1.set_xlabel('Probability')
        ax1.set_title('Top Predicted Outcomes')
        
        # 2. Feature Importance (if available)
        ax2 = axes[0, 1]
        if 'feature_importance' in explanation_results['explanations']:
            feature_exp = explanation_results['explanations']['feature_importance']
            if 'sorted_features' in feature_exp:
                top_features = feature_exp['sorted_features'][:10]
                if top_features:
                    feature_names, importances = zip(*[(name, data['combined_importance']) 
                                                     for name, data in top_features])
                    
                    ax2.barh(range(len(feature_names)), importances)
                    ax2.set_yticks(range(len(feature_names)))
                    ax2.set_yticklabels(feature_names)
                    ax2.set_xlabel('Importance Score')
                    ax2.set_title('Top Feature Importances')
                else:
                    ax2.text(0.5, 0.5, 'No features available', ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'Feature importance\nnot calculated', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
        
        # 3. Integrated Gradients Summary
        ax3 = axes[0, 2]
        if 'integrated_gradients' in explanation_results['explanations']:
            ig_exp = explanation_results['explanations']['integrated_gradients']
            if 'attribution_summary' in ig_exp:
                summary = ig_exp['attribution_summary']
                categories = ['Text Features', 'Numerical Features']
                importances = [summary['text_importance'], summary['numerical_importance']]
                
                ax3.bar(categories, importances)
                ax3.set_ylabel('Attribution Magnitude')
                ax3.set_title('Attribution by Feature Type')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'IG attribution\nerror occurred', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'Integrated Gradients\nnot available', ha='center', va='center')
        
        # 4. Trading Signal Analysis
        ax4 = axes[1, 0]
        signal_data = {
            'Signal': prediction['trading_signal'],
            'Strength': prediction['signal_strength'],
            'Price Confidence': prediction['price_confidence'],
            'Outcome Confidence': prediction['outcome_confidence']
        }
        
        ax4.axis('off')
        signal_text = '\n'.join([f'{k}: {v:.3f}' if isinstance(v, (int, float)) 
                               else f'{k}: {v}' for k, v in signal_data.items()])
        ax4.text(0.1, 0.9, 'Trading Signal Analysis:\n\n' + signal_text, 
                transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                fontfamily='monospace')
        
        # 5. Text Processing Summary
        ax5 = axes[1, 1]
        text_summary = f"""Text Processing Summary:
        
Original length: {len(processed_play.original_text)}
Cleaned length: {len(processed_play.cleaned_text)}
Players found: {len(processed_play.entities.players)}
Actions found: {len(processed_play.entities.actions)}
Confidence: {processed_play.confidence:.3f}

Entities:
Players: {', '.join(processed_play.entities.players[:3])}
Actions: {', '.join(processed_play.entities.actions)}
Yardage: {processed_play.entities.yardage}
"""
        
        ax5.axis('off')
        ax5.text(0.1, 0.9, text_summary, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 6. Model Predictions
        ax6 = axes[1, 2]
        pred_summary = f"""Model Predictions:
        
Predicted Outcome: {prediction['predicted_outcome']}
Outcome Confidence: {prediction['outcome_confidence']:.3f}

Price Change: {prediction['predicted_price_change']:.4f}
Price Confidence: {prediction['price_confidence']:.3f}
Price Volatility: {prediction['price_volatility']:.4f}

Processing Time: {prediction['processing_time']:.4f}s
"""
        
        ax6.axis('off')
        ax6.text(0.1, 0.9, pred_summary, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        safe_filename = "".join(c for c in play_description[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plot_path = self.output_dir / f"explanation_{safe_filename}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Explanation plot saved to {plot_path}")
    
    def _save_global_importance_results(self, results: Dict[str, Any]):
        """Save global feature importance results"""
        
        import json
        
        # Save JSON results
        results_path = self.output_dir / "global_feature_importance.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create importance plot
        plt.figure(figsize=(12, 8))
        
        if results['feature_ranking']:
            features, importances = zip(*results['feature_ranking'][:15])  # Top 15
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance Score')
            plt.title('Global Feature Importance Ranking')
            plt.gca().invert_yaxis()
            
            # Add error bars if std available
            if 'numerical_importance_std' in results:
                stds = [results['numerical_importance_std'][results['feature_names'].index(feat)] 
                       for feat in features if feat in results['feature_names']]
                if len(stds) == len(importances):
                    plt.errorbar(importances, range(len(features)), xerr=stds, 
                               fmt='none', capsize=3, alpha=0.7)
        
        plt.tight_layout()
        plot_path = self.output_dir / "global_feature_importance.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Global importance results saved to {results_path}")
        logger.info(f"Global importance plot saved to {plot_path}")
    
    def create_model_summary_report(self) -> str:
        """Create a summary report of model explainability"""
        
        report = f"""
NFL Play Analysis Model - Explainability Summary
==============================================

Model Architecture:
- Text embedding dimension: {self.model.config.text_embedding_dim}
- Numerical feature dimension: {self.model.config.numerical_feature_dim}
- Hidden dimension: {self.model.config.hidden_dim}
- Number of attention heads: {self.model.config.num_attention_heads}
- Number of transformer layers: {self.model.config.num_transformer_layers}

Available Explanation Methods:
- Attention Weight Analysis ✓
- Integrated Gradients ✓
- Feature Importance (Occlusion) ✓
- Gradient SHAP ✓

Text Processing Features:
- Entity extraction (players, actions, outcomes)
- Play normalization and cleaning
- Confidence scoring
- Embedding generation

Model Interpretability:
- Individual play explanations
- Global feature importance ranking
- Attribution visualization
- Trading signal explanation

Usage:
1. Use analyze_play_explanation() for individual plays
2. Use analyze_feature_importance_global() for overall patterns
3. Check attention weights for transformer interpretability
4. Review feature importance for numerical feature relevance
"""
        
        # Save report
        report_path = self.output_dir / "explainability_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report