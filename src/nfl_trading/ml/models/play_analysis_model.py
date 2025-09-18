"""
Play Analysis Model

Multi-task transformer model for NFL play outcome prediction and price impact estimation.
Optimized for free GPU instances with memory-efficient training.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for PlayAnalysisModel"""
    
    # Model architecture
    text_embedding_dim: int = 768
    numerical_feature_dim: int = 25
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_transformer_layers: int = 4
    dropout_rate: float = 0.1
    
    # Task configuration
    num_outcome_classes: int = 8  # Different play outcomes
    price_impact_dim: int = 1     # Single price change prediction
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Model capacity
    max_sequence_length: int = 128
    
    # Loss weighting
    outcome_loss_weight: float = 1.0
    price_loss_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text_embedding_dim': self.text_embedding_dim,
            'numerical_feature_dim': self.numerical_feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_attention_heads': self.num_attention_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'dropout_rate': self.dropout_rate,
            'num_outcome_classes': self.num_outcome_classes,
            'price_impact_dim': self.price_impact_dim,
            'gradient_checkpointing': self.gradient_checkpointing,
            'mixed_precision': self.mixed_precision,
            'max_sequence_length': self.max_sequence_length,
            'outcome_loss_weight': self.outcome_loss_weight,
            'price_loss_weight': self.price_loss_weight
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional masking"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention, attention_weights = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention)
        
        return output, attention_weights
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention = torch.matmul(attention_weights, V)
        
        return attention, attention_weights


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class FeatureFusion(nn.Module):
    """Fuse text embeddings with numerical features"""
    
    def __init__(self, text_dim: int, numerical_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.text_projection = nn.Linear(text_dim, output_dim // 2)
        self.numerical_projection = nn.Linear(numerical_dim, output_dim // 2)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, text_features: torch.Tensor, numerical_features: torch.Tensor) -> torch.Tensor:
        # Project features to same dimension
        text_proj = self.text_projection(text_features)
        numerical_proj = self.numerical_projection(numerical_features)
        
        # Concatenate and fuse
        fused = torch.cat([text_proj, numerical_proj], dim=-1)
        fused = self.fusion_layer(fused)
        fused = self.layer_norm(fused)
        
        return fused


class PlayAnalysisModel(nn.Module):
    """
    Multi-task transformer model for NFL play analysis.
    
    Features:
    - Multi-task learning: play outcome classification + price impact regression
    - Text and numerical feature fusion
    - Attention mechanism for interpretability
    - Memory-efficient design for free GPU instances
    - Gradient checkpointing and mixed precision support
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(
            text_dim=config.text_embedding_dim,
            numerical_dim=config.numerical_feature_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout_rate
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_dim,
            max_len=config.max_sequence_length
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.hidden_dim,
                num_heads=config.num_attention_heads,
                d_ff=config.hidden_dim * 4,
                dropout=config.dropout_rate
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        # Task-specific heads
        self.outcome_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_outcome_classes)
        )
        
        self.price_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.price_impact_dim)
        )
        
        # Attention weights storage for interpretability
        self.attention_weights = []
        
        # Initialize weights
        self._init_weights()
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.transformer_layers:
            layer = torch.utils.checkpoint.checkpoint(layer)
    
    def forward(self, 
                text_embeddings: torch.Tensor,
                numerical_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_embeddings: [batch_size, seq_len, text_embedding_dim]
            numerical_features: [batch_size, numerical_feature_dim]
            attention_mask: [batch_size, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with 'outcome_logits', 'price_prediction', and optionally 'attention_weights'
        """
        
        batch_size, seq_len = text_embeddings.shape[:2]
        
        # Expand numerical features to match sequence length
        numerical_expanded = numerical_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Feature fusion
        fused_features = self.feature_fusion(text_embeddings, numerical_expanded)
        
        # Add positional encoding
        fused_features = fused_features.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        fused_features = self.positional_encoding(fused_features)
        fused_features = fused_features.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer layers
        hidden_states = fused_features
        all_attention_weights = []
        
        for layer in self.transformer_layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states, attention_weights = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask
                )
            else:
                hidden_states, attention_weights = layer(hidden_states, attention_mask)
            
            if return_attention:
                all_attention_weights.append(attention_weights)
        
        # Global average pooling (attention-weighted)
        if attention_mask is not None:
            # Mask padded positions
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask_expanded
            pooled_features = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_features = hidden_states.mean(dim=1)
        
        # Task-specific predictions
        outcome_logits = self.outcome_classifier(pooled_features)
        price_prediction = self.price_predictor(pooled_features)
        
        outputs = {
            'outcome_logits': outcome_logits,
            'price_prediction': price_prediction
        }
        
        if return_attention:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from last forward pass"""
        return self.attention_weights
    
    def get_feature_importance(self, 
                             text_embeddings: torch.Tensor,
                             numerical_features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get feature importance using integrated gradients
        
        Returns:
            Dictionary with importance scores for text and numerical features
        """
        
        self.eval()
        
        # Enable gradients for input
        text_embeddings.requires_grad_(True)
        numerical_features.requires_grad_(True)
        
        # Forward pass
        outputs = self.forward(text_embeddings, numerical_features, attention_mask)
        
        # Get gradients for both tasks
        outcome_importance = torch.autograd.grad(
            outputs['outcome_logits'].sum(), 
            [text_embeddings, numerical_features],
            retain_graph=True,
            create_graph=False
        )
        
        price_importance = torch.autograd.grad(
            outputs['price_prediction'].sum(),
            [text_embeddings, numerical_features],
            retain_graph=False,
            create_graph=False
        )
        
        return {
            'text_importance_outcome': outcome_importance[0],
            'numerical_importance_outcome': outcome_importance[1],
            'text_importance_price': price_importance[0],
            'numerical_importance_price': price_importance[1]
        }
    
    def predict_play_impact(self, 
                           text_embeddings: torch.Tensor,
                           numerical_features: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           return_confidence: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict play impact with confidence scores
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(text_embeddings, numerical_features, attention_mask)
            
            # Convert logits to probabilities
            outcome_probs = F.softmax(outputs['outcome_logits'], dim=-1)
            
            # Get confidence scores
            if return_confidence:
                outcome_confidence = outcome_probs.max(dim=-1)[0]
                price_confidence = torch.ones_like(outputs['price_prediction'].squeeze())  # Placeholder
                
                outputs['outcome_confidence'] = outcome_confidence
                outputs['price_confidence'] = price_confidence
            
            outputs['outcome_probs'] = outcome_probs
        
        return outputs
    
    def save_model(self, filepath: str):
        """Save model with config"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'PlayAnalysisModel':
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=device)
        config = ModelConfig.from_dict(checkpoint['config'])
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss function with automatic weighting"""
    
    def __init__(self, 
                 outcome_weight: float = 1.0, 
                 price_weight: float = 1.0,
                 adaptive_weighting: bool = True):
        super().__init__()
        
        self.outcome_weight = outcome_weight
        self.price_weight = price_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Loss functions
        self.outcome_loss_fn = nn.CrossEntropyLoss()
        self.price_loss_fn = nn.MSELoss()
        
        # Adaptive weighting parameters
        if adaptive_weighting:
            self.log_vars = nn.Parameter(torch.zeros(2))
    
    def forward(self, 
                outcome_logits: torch.Tensor,
                price_predictions: torch.Tensor,
                outcome_targets: torch.Tensor,
                price_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Returns:
            Dictionary with total loss and individual losses
        """
        
        # Individual losses
        outcome_loss = self.outcome_loss_fn(outcome_logits, outcome_targets)
        price_loss = self.price_loss_fn(price_predictions.squeeze(), price_targets)
        
        if self.adaptive_weighting:
            # Adaptive weighting based on uncertainty
            precision1 = torch.exp(-self.log_vars[0])
            precision2 = torch.exp(-self.log_vars[1])
            
            total_loss = (
                precision1 * outcome_loss + self.log_vars[0] +
                precision2 * price_loss + self.log_vars[1]
            )
        else:
            # Fixed weighting
            total_loss = self.outcome_weight * outcome_loss + self.price_weight * price_loss
        
        return {
            'total_loss': total_loss,
            'outcome_loss': outcome_loss,
            'price_loss': price_loss,
            'outcome_weight': precision1 if self.adaptive_weighting else self.outcome_weight,
            'price_weight': precision2 if self.adaptive_weighting else self.price_weight
        }