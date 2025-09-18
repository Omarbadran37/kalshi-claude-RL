"""
Play Predictor

Real-time inference engine for NFL play analysis with uncertainty quantification
and trading signal generation.
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pickle

from ..models.play_analysis_model import PlayAnalysisModel, ModelConfig
from ..text_processor import PlayTextProcessor, ProcessedPlay

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Trading signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class PredictionResult:
    """Result from play prediction"""
    
    # Play outcome prediction
    outcome_probabilities: Dict[str, float]
    predicted_outcome: str
    outcome_confidence: float
    
    # Price impact prediction
    predicted_price_change: float
    price_confidence: float
    price_volatility: float
    
    # Trading signals
    trading_signal: TradingSignal
    signal_strength: float
    
    # Model internals
    attention_weights: Optional[torch.Tensor]
    feature_importance: Optional[Dict[str, float]]
    
    # Processing info
    processing_time: float
    model_version: str
    
    # Raw outputs
    raw_outcome_logits: torch.Tensor
    raw_price_prediction: torch.Tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'outcome_probabilities': self.outcome_probabilities,
            'predicted_outcome': self.predicted_outcome,
            'outcome_confidence': self.outcome_confidence,
            'predicted_price_change': self.predicted_price_change,
            'price_confidence': self.price_confidence,
            'price_volatility': self.price_volatility,
            'trading_signal': self.trading_signal.value,
            'signal_strength': self.signal_strength,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'feature_importance': self.feature_importance
        }


class PlayPredictor:
    """
    Real-time play prediction engine
    
    Features:
    - Fast inference with caching
    - Uncertainty quantification 
    - Trading signal generation
    - Feature importance analysis
    - Batch prediction support
    - Memory-efficient operation
    """
    
    def __init__(self,
                 model: PlayAnalysisModel,
                 text_processor: PlayTextProcessor,
                 numerical_scaler: Optional[StandardScaler] = None,
                 outcome_labels: Optional[List[str]] = None,
                 device: str = "auto",
                 enable_cache: bool = True,
                 max_cache_size: int = 1000):
        
        self.model = model
        self.text_processor = text_processor
        self.numerical_scaler = numerical_scaler
        self.outcome_labels = outcome_labels or [f"outcome_{i}" for i in range(model.config.num_outcome_classes)]
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Caching
        self.enable_cache = enable_cache
        self.prediction_cache = {} if enable_cache else None
        self.max_cache_size = max_cache_size
        
        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
        # Trading signal thresholds
        self.signal_thresholds = {
            'strong_buy': 0.03,
            'buy': 0.01,
            'sell': -0.01,
            'strong_sell': -0.03
        }
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.6
        self.high_confidence_threshold = 0.8
        
        logger.info(f"PlayPredictor initialized on {self.device}")
    
    @classmethod
    def from_checkpoint(cls,
                       checkpoint_path: str,
                       text_processor: PlayTextProcessor,
                       preprocessor_path: Optional[str] = None,
                       device: str = "auto") -> 'PlayPredictor':
        """
        Load predictor from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            text_processor: Text processor instance
            preprocessor_path: Path to preprocessor objects
            device: Device to use
        
        Returns:
            PlayPredictor instance
        """
        
        logger.info(f"Loading predictor from checkpoint: {checkpoint_path}")
        
        # Load model
        model = PlayAnalysisModel.load_model(checkpoint_path, device)
        
        # Load preprocessors if available
        numerical_scaler = None
        outcome_labels = None
        
        if preprocessor_path:
            try:
                with open(preprocessor_path, 'rb') as f:
                    preprocessors = pickle.load(f)
                    numerical_scaler = preprocessors.get('numerical_scaler')
                    outcome_encoder = preprocessors.get('outcome_encoder')
                    
                    if outcome_encoder:
                        outcome_labels = outcome_encoder.classes_.tolist()
                        
            except Exception as e:
                logger.warning(f"Could not load preprocessors: {e}")
        
        return cls(
            model=model,
            text_processor=text_processor,
            numerical_scaler=numerical_scaler,
            outcome_labels=outcome_labels,
            device=device
        )
    
    def predict_play(self,
                    play_description: str,
                    numerical_features: Optional[Dict[str, float]] = None,
                    return_feature_importance: bool = False,
                    return_attention: bool = False) -> PredictionResult:
        """
        Predict outcome and price impact for a single play
        
        Args:
            play_description: Text description of the play
            numerical_features: Dictionary of numerical game features
            return_feature_importance: Whether to compute feature importance
            return_attention: Whether to return attention weights
        
        Returns:
            PredictionResult object
        """
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(play_description, numerical_features)
        if self.enable_cache and cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        # Process text
        processed_play = self.text_processor.process_play(play_description)
        
        if processed_play.embeddings is None:
            # Return default prediction for failed processing
            return self._create_default_prediction(start_time)
        
        # Prepare numerical features
        numerical_tensor = self._prepare_numerical_features(
            processed_play, numerical_features
        )
        
        # Prepare inputs
        text_embedding = processed_play.embeddings.unsqueeze(0).to(self.device)
        numerical_tensor = numerical_tensor.unsqueeze(0).to(self.device)
        
        # Create attention mask
        attention_mask = torch.ones(1, 1).to(self.device)  # Single sequence
        
        # Model prediction
        with torch.no_grad():
            outputs = self.model(
                text_embedding,
                numerical_tensor,
                attention_mask,
                return_attention=return_attention
            )
        
        # Process outputs
        outcome_logits = outputs['outcome_logits']
        price_prediction = outputs['price_prediction']
        
        # Calculate probabilities and confidence
        outcome_probs = F.softmax(outcome_logits, dim=-1)
        outcome_probabilities = {
            self.outcome_labels[i]: prob.item() 
            for i, prob in enumerate(outcome_probs[0])
        }
        
        predicted_outcome_idx = outcome_probs.argmax().item()
        predicted_outcome = self.outcome_labels[predicted_outcome_idx]
        outcome_confidence = outcome_probs.max().item()
        
        # Price prediction
        predicted_price_change = price_prediction.item()
        
        # Calculate price confidence using model uncertainty
        price_confidence = self._estimate_price_confidence(
            price_prediction, processed_play.confidence
        )
        
        # Estimate price volatility
        price_volatility = self._estimate_price_volatility(processed_play)
        
        # Generate trading signal
        trading_signal, signal_strength = self._generate_trading_signal(
            predicted_price_change, outcome_confidence, price_confidence
        )
        
        # Feature importance (optional)
        feature_importance = None
        if return_feature_importance:
            feature_importance = self._compute_feature_importance(
                text_embedding, numerical_tensor, attention_mask
            )
        
        # Attention weights (optional)
        attention_weights = None
        if return_attention and 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
        
        # Create result
        result = PredictionResult(
            outcome_probabilities=outcome_probabilities,
            predicted_outcome=predicted_outcome,
            outcome_confidence=outcome_confidence,
            predicted_price_change=predicted_price_change,
            price_confidence=price_confidence,
            price_volatility=price_volatility,
            trading_signal=trading_signal,
            signal_strength=signal_strength,
            attention_weights=attention_weights,
            feature_importance=feature_importance,
            processing_time=time.time() - start_time,
            model_version=getattr(self.model, 'version', 'unknown'),
            raw_outcome_logits=outcome_logits,
            raw_price_prediction=price_prediction
        )
        
        # Cache result
        if self.enable_cache:
            self._cache_prediction(cache_key, result)
        
        # Update statistics
        self.prediction_count += 1
        self.total_inference_time += result.processing_time
        
        return result
    
    def predict_batch(self,
                     play_descriptions: List[str],
                     numerical_features_list: Optional[List[Dict[str, float]]] = None,
                     batch_size: int = 16) -> List[PredictionResult]:
        """
        Predict for multiple plays efficiently
        
        Args:
            play_descriptions: List of play descriptions
            numerical_features_list: List of numerical feature dictionaries
            batch_size: Batch size for processing
        
        Returns:
            List of PredictionResult objects
        """
        
        logger.info(f"Predicting batch of {len(play_descriptions)} plays")
        
        if numerical_features_list is None:
            numerical_features_list = [None] * len(play_descriptions)
        
        results = []
        
        # Process in batches
        for i in range(0, len(play_descriptions), batch_size):
            batch_descriptions = play_descriptions[i:i + batch_size]
            batch_numerical = numerical_features_list[i:i + batch_size]
            
            batch_results = self._predict_batch_internal(
                batch_descriptions, batch_numerical
            )
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self,
                               descriptions: List[str],
                               numerical_features_list: List[Optional[Dict]]) -> List[PredictionResult]:
        """Internal batch prediction"""
        
        start_time = time.time()
        
        # Process all texts
        processed_plays = self.text_processor.process_plays(descriptions)
        
        # Filter valid plays
        valid_indices = []
        valid_plays = []
        for i, play in enumerate(processed_plays):
            if play.embeddings is not None:
                valid_indices.append(i)
                valid_plays.append(play)
        
        if not valid_plays:
            # Return default predictions for all
            return [self._create_default_prediction(start_time) for _ in descriptions]
        
        # Prepare batch tensors
        text_embeddings = torch.stack([play.embeddings for play in valid_plays])
        
        numerical_tensors = []
        for i in valid_indices:
            numerical_tensor = self._prepare_numerical_features(
                processed_plays[i], numerical_features_list[i]
            )
            numerical_tensors.append(numerical_tensor)
        
        numerical_batch = torch.stack(numerical_tensors)
        
        # Move to device
        text_embeddings = text_embeddings.to(self.device)
        numerical_batch = numerical_batch.to(self.device)
        
        # Create attention masks
        attention_masks = torch.ones(len(valid_plays), 1).to(self.device)
        
        # Batch prediction
        with torch.no_grad():
            outputs = self.model(
                text_embeddings,
                numerical_batch,
                attention_masks
            )
        
        # Process outputs
        outcome_logits = outputs['outcome_logits']
        price_predictions = outputs['price_prediction']
        
        # Create results for valid predictions
        valid_results = []
        for i, (play_idx, processed_play) in enumerate(zip(valid_indices, valid_plays)):
            # Extract individual prediction
            outcome_logit = outcome_logits[i:i+1]
            price_pred = price_predictions[i:i+1]
            
            # Process same as single prediction
            outcome_probs = F.softmax(outcome_logit, dim=-1)
            outcome_probabilities = {
                self.outcome_labels[j]: prob.item() 
                for j, prob in enumerate(outcome_probs[0])
            }
            
            predicted_outcome_idx = outcome_probs.argmax().item()
            predicted_outcome = self.outcome_labels[predicted_outcome_idx]
            outcome_confidence = outcome_probs.max().item()
            
            predicted_price_change = price_pred.item()
            price_confidence = self._estimate_price_confidence(
                price_pred, processed_play.confidence
            )
            price_volatility = self._estimate_price_volatility(processed_play)
            
            trading_signal, signal_strength = self._generate_trading_signal(
                predicted_price_change, outcome_confidence, price_confidence
            )
            
            result = PredictionResult(
                outcome_probabilities=outcome_probabilities,
                predicted_outcome=predicted_outcome,
                outcome_confidence=outcome_confidence,
                predicted_price_change=predicted_price_change,
                price_confidence=price_confidence,
                price_volatility=price_volatility,
                trading_signal=trading_signal,
                signal_strength=signal_strength,
                attention_weights=None,
                feature_importance=None,
                processing_time=time.time() - start_time,
                model_version=getattr(self.model, 'version', 'unknown'),
                raw_outcome_logits=outcome_logit,
                raw_price_prediction=price_pred
            )
            
            valid_results.append(result)
        
        # Create final results list with default predictions for invalid plays
        all_results = []
        valid_iter = iter(valid_results)
        
        for i in range(len(descriptions)):
            if i in valid_indices:
                all_results.append(next(valid_iter))
            else:
                all_results.append(self._create_default_prediction(start_time))
        
        return all_results
    
    def _prepare_numerical_features(self,
                                   processed_play: ProcessedPlay,
                                   additional_features: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Prepare numerical features tensor"""
        
        # Extract features from processed play
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
            # Add zeros for missing additional features
            features.extend([0.0] * 9)
        
        # Action type one-hot encoding
        action_types = ['pass', 'rush', 'kick', 'turnover', 'penalty', 'score']
        for action_type in action_types:
            features.append(1.0 if action_type in processed_play.entities.actions else 0.0)
        
        # Outcome type one-hot encoding
        outcome_types = ['success', 'failure', 'neutral']
        for outcome_type in outcome_types:
            features.append(1.0 if outcome_type in processed_play.entities.outcomes else 0.0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Apply scaling if available
        if self.numerical_scaler is not None:
            features_array = features_tensor.numpy().reshape(1, -1)
            scaled_features = self.numerical_scaler.transform(features_array)
            features_tensor = torch.tensor(scaled_features[0], dtype=torch.float32)
        
        return features_tensor
    
    def _estimate_price_confidence(self, price_prediction: torch.Tensor, text_confidence: float) -> float:
        """Estimate confidence in price prediction"""
        
        # Use text processing confidence as base
        base_confidence = text_confidence
        
        # Adjust based on prediction magnitude (more extreme predictions are less confident)
        magnitude = abs(price_prediction.item())
        magnitude_penalty = min(magnitude * 5, 0.3)  # Cap penalty
        
        confidence = base_confidence - magnitude_penalty
        return max(0.1, min(1.0, confidence))
    
    def _estimate_price_volatility(self, processed_play: ProcessedPlay) -> float:
        """Estimate price volatility based on play characteristics"""
        
        volatility = 0.02  # Base volatility
        
        # High impact events increase volatility
        if 'score' in processed_play.entities.actions:
            volatility += 0.03
        if 'turnover' in processed_play.entities.actions:
            volatility += 0.02
        if processed_play.entities.score_change and processed_play.entities.score_change > 0:
            volatility += 0.02
        
        # High yardage plays increase volatility
        if processed_play.entities.yardage and abs(processed_play.entities.yardage) > 20:
            volatility += 0.01
        
        return min(volatility, 0.1)  # Cap volatility
    
    def _generate_trading_signal(self,
                                price_change: float,
                                outcome_confidence: float,
                                price_confidence: float) -> Tuple[TradingSignal, float]:
        """Generate trading signal based on predictions"""
        
        # Overall confidence
        overall_confidence = (outcome_confidence + price_confidence) / 2
        
        # Only generate signals above minimum confidence
        if overall_confidence < self.min_confidence_threshold:
            return TradingSignal.HOLD, 0.0
        
        # Determine signal based on price change and confidence
        signal_strength = abs(price_change) * overall_confidence
        
        if price_change >= self.signal_thresholds['strong_buy']:
            if overall_confidence > self.high_confidence_threshold:
                return TradingSignal.STRONG_BUY, signal_strength
            else:
                return TradingSignal.BUY, signal_strength
        elif price_change >= self.signal_thresholds['buy']:
            return TradingSignal.BUY, signal_strength
        elif price_change <= self.signal_thresholds['strong_sell']:
            if overall_confidence > self.high_confidence_threshold:
                return TradingSignal.STRONG_SELL, signal_strength
            else:
                return TradingSignal.SELL, signal_strength
        elif price_change <= self.signal_thresholds['sell']:
            return TradingSignal.SELL, signal_strength
        else:
            return TradingSignal.HOLD, 0.0
    
    def _compute_feature_importance(self,
                                   text_embedding: torch.Tensor,
                                   numerical_features: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Dict[str, float]:
        """Compute feature importance using gradients"""
        
        # Enable gradients
        text_embedding.requires_grad_(True)
        numerical_features.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(text_embedding, numerical_features, attention_mask)
        
        # Calculate gradients for both tasks
        price_grad = torch.autograd.grad(
            outputs['price_prediction'].sum(),
            [text_embedding, numerical_features],
            retain_graph=True,
            create_graph=False
        )
        
        outcome_grad = torch.autograd.grad(
            outputs['outcome_logits'].sum(),
            [text_embedding, numerical_features],
            retain_graph=False,
            create_graph=False
        )
        
        # Calculate importance scores
        text_importance = torch.norm(price_grad[0]).item() + torch.norm(outcome_grad[0]).item()
        numerical_importance = torch.norm(price_grad[1]).item() + torch.norm(outcome_grad[1]).item()
        
        total_importance = text_importance + numerical_importance
        
        if total_importance > 0:
            return {
                'text_features': text_importance / total_importance,
                'numerical_features': numerical_importance / total_importance
            }
        else:
            return {'text_features': 0.5, 'numerical_features': 0.5}
    
    def _create_default_prediction(self, start_time: float) -> PredictionResult:
        """Create default prediction for failed processing"""
        
        return PredictionResult(
            outcome_probabilities={label: 1.0 / len(self.outcome_labels) for label in self.outcome_labels},
            predicted_outcome=self.outcome_labels[0],
            outcome_confidence=0.0,
            predicted_price_change=0.0,
            price_confidence=0.0,
            price_volatility=0.05,
            trading_signal=TradingSignal.HOLD,
            signal_strength=0.0,
            attention_weights=None,
            feature_importance=None,
            processing_time=time.time() - start_time,
            model_version=getattr(self.model, 'version', 'unknown'),
            raw_outcome_logits=torch.zeros(1, len(self.outcome_labels)),
            raw_price_prediction=torch.zeros(1, 1)
        )
    
    def _get_cache_key(self, play_description: str, numerical_features: Optional[Dict]) -> str:
        """Generate cache key"""
        import hashlib
        
        content = play_description
        if numerical_features:
            content += str(sorted(numerical_features.items()))
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        
        if len(self.prediction_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        # Store without tensors to save memory
        cached_result = PredictionResult(
            outcome_probabilities=result.outcome_probabilities,
            predicted_outcome=result.predicted_outcome,
            outcome_confidence=result.outcome_confidence,
            predicted_price_change=result.predicted_price_change,
            price_confidence=result.price_confidence,
            price_volatility=result.price_volatility,
            trading_signal=result.trading_signal,
            signal_strength=result.signal_strength,
            attention_weights=None,  # Don't cache tensors
            feature_importance=result.feature_importance,
            processing_time=result.processing_time,
            model_version=result.model_version,
            raw_outcome_logits=torch.zeros(1, len(self.outcome_labels)),
            raw_price_prediction=torch.zeros(1, 1)
        )
        
        self.prediction_cache[cache_key] = cached_result
    
    def set_signal_thresholds(self, thresholds: Dict[str, float]):
        """Update trading signal thresholds"""
        self.signal_thresholds.update(thresholds)
        logger.info(f"Updated signal thresholds: {self.signal_thresholds}")
    
    def set_confidence_thresholds(self, min_confidence: float, high_confidence: float):
        """Update confidence thresholds"""
        self.min_confidence_threshold = min_confidence
        self.high_confidence_threshold = high_confidence
        logger.info(f"Updated confidence thresholds: min={min_confidence}, high={high_confidence}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        
        avg_inference_time = (
            self.total_inference_time / self.prediction_count 
            if self.prediction_count > 0 else 0.0
        )
        
        return {
            'device': str(self.device),
            'prediction_count': self.prediction_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': avg_inference_time,
            'cache_size': len(self.prediction_cache) if self.prediction_cache else 0,
            'cache_enabled': self.enable_cache,
            'model_config': self.model.config.to_dict(),
            'signal_thresholds': self.signal_thresholds,
            'confidence_thresholds': {
                'min_confidence': self.min_confidence_threshold,
                'high_confidence': self.high_confidence_threshold
            }
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        if self.prediction_cache:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")
    
    def enable_benchmarking(self):
        """Enable benchmarking mode for performance testing"""
        self.enable_cache = False
        logger.info("Benchmarking mode enabled - cache disabled")
    
    def save_state(self, filepath: str):
        """Save predictor state"""
        state = {
            'signal_thresholds': self.signal_thresholds,
            'confidence_thresholds': {
                'min_confidence': self.min_confidence_threshold,
                'high_confidence': self.high_confidence_threshold
            },
            'stats': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            import json
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Predictor state saved to {filepath}")