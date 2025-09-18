"""
Model Trainer

Comprehensive training pipeline for NFL play analysis models with GPU optimization
for free tier instances, hyperparameter tuning, and advanced training techniques.
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)
import optuna
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.play_analysis_model import PlayAnalysisModel, ModelConfig, MultiTaskLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'plateau', 'onecycle', 'none'
    warmup_steps: int = 100
    
    # Memory optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    accumulate_grad_batches: int = 1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    monitor_metric: str = 'val_total_loss'  # metric to monitor for early stopping
    
    # Checkpointing
    save_top_k: int = 3
    save_every_n_epochs: int = 5
    
    # Loss weighting
    outcome_loss_weight: float = 1.0
    price_loss_weight: float = 1.0
    adaptive_loss_weighting: bool = True
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Logging
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class ModelTrainer:
    """
    Comprehensive model trainer for NFL play analysis
    
    Features:
    - Mixed precision training for memory efficiency
    - Advanced learning rate scheduling
    - Early stopping and checkpointing
    - Hyperparameter optimization with Optuna
    - Comprehensive metrics tracking
    - GPU memory optimization for free tiers
    """
    
    def __init__(self,
                 model: PlayAnalysisModel,
                 config: TrainingConfig,
                 device: Optional[str] = None,
                 experiment_dir: str = "experiments",
                 experiment_name: Optional[str] = None):
        
        self.model = model
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = MultiTaskLoss(
            outcome_weight=config.outcome_loss_weight,
            price_weight=config.price_loss_weight,
            adaptive_weighting=config.adaptive_loss_weighting
        ).to(self.device)
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'
        )
        
        # Tracking
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.current_epoch = 0
        self.global_step = 0
        
        # Create experiment directory
        self.exp_path = self.experiment_dir / self.experiment_name
        self.exp_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
        
        logger.info(f"Trainer initialized for experiment: {self.experiment_name}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        
        if self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler"""
        
        if self.config.scheduler.lower() == 'none':
            return None
        elif self.config.scheduler.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config.scheduler.lower() == 'onecycle':
            # Will be set when we know the number of steps
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _save_config(self):
        """Save training configuration"""
        config_path = self.exp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'training_config': self.config.to_dict(),
                'model_config': self.model.config.to_dict()
            }, f, indent=2)
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
        
        Returns:
            Dictionary with training results
        """
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Setup OneCycle scheduler if needed
        if self.config.scheduler.lower() == 'onecycle':
            total_steps = len(train_loader) * self.config.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.config.validate_every_n_epochs == 0:
                val_metrics = self._validate_epoch(val_loader)
                
                # Update metrics history
                for key, value in train_metrics.items():
                    self.metrics_history[f'train_{key}'].append(value)
                for key, value in val_metrics.items():
                    self.metrics_history[f'val_{key}'].append(value)
                
                # Log metrics
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Learning rate scheduling
                if self.scheduler is not None and self.config.scheduler.lower() == 'plateau':
                    self.scheduler.step(val_metrics['total_loss'])
                elif self.scheduler is not None and self.config.scheduler.lower() == 'cosine':
                    self.scheduler.step()
                
                # Early stopping check
                monitor_value = val_metrics.get(
                    self.config.monitor_metric.replace('val_', ''), 
                    val_metrics['total_loss']
                )
                
                if self.early_stopping(monitor_value):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Save checkpoint
                if self._should_save_checkpoint(epoch, val_metrics):
                    self._save_checkpoint(epoch, val_metrics)
            
            # Save periodic checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, {}, prefix='periodic')
        
        # Final evaluation
        results = self._finalize_training(train_loader, val_loader, test_loader, start_time)
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        
        total_loss = 0.0
        total_outcome_loss = 0.0
        total_price_loss = 0.0
        num_batches = 0
        
        outcome_preds = []
        outcome_targets = []
        price_preds = []
        price_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(
                    batch['text_embedding'],
                    batch['numerical_features'],
                    batch.get('attention_mask')
                )
                
                loss_dict = self.criterion(
                    outputs['outcome_logits'],
                    outputs['price_prediction'],
                    batch['outcome_label'],
                    batch['price_target']
                )
                
                loss = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                if self.config.accumulate_grad_batches > 1:
                    loss = loss / self.config.accumulate_grad_batches
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
                
                if self.scaler is not None:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    
                    self.optimizer.step()
                
                # OneCycle scheduler step
                if (self.scheduler is not None and 
                    self.config.scheduler.lower() == 'onecycle'):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_outcome_loss += loss_dict['outcome_loss'].item()
            total_price_loss += loss_dict['price_loss'].item()
            num_batches += 1
            
            # Store predictions for metrics
            outcome_preds.extend(outputs['outcome_logits'].argmax(dim=1).cpu().numpy())
            outcome_targets.extend(batch['outcome_label'].cpu().numpy())
            price_preds.extend(outputs['price_prediction'].squeeze().cpu().numpy())
            price_targets.extend(batch['price_target'].cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
            
            # Memory cleanup for free tiers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_outcome_loss = total_outcome_loss / num_batches
        avg_price_loss = total_price_loss / num_batches
        
        # Classification metrics
        outcome_accuracy = accuracy_score(outcome_targets, outcome_preds)
        
        # Regression metrics
        price_mse = mean_squared_error(price_targets, price_preds)
        price_mae = mean_absolute_error(price_targets, price_preds)
        
        return {
            'total_loss': avg_loss,
            'outcome_loss': avg_outcome_loss,
            'price_loss': avg_price_loss,
            'outcome_accuracy': outcome_accuracy,
            'price_mse': price_mse,
            'price_mae': price_mae
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        
        total_loss = 0.0
        total_outcome_loss = 0.0
        total_price_loss = 0.0
        num_batches = 0
        
        outcome_preds = []
        outcome_targets = []
        price_preds = []
        price_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(
                        batch['text_embedding'],
                        batch['numerical_features'],
                        batch.get('attention_mask')
                    )
                    
                    loss_dict = self.criterion(
                        outputs['outcome_logits'],
                        outputs['price_prediction'],
                        batch['outcome_label'],
                        batch['price_target']
                    )
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item()
                total_outcome_loss += loss_dict['outcome_loss'].item()
                total_price_loss += loss_dict['price_loss'].item()
                num_batches += 1
                
                # Store predictions
                outcome_preds.extend(outputs['outcome_logits'].argmax(dim=1).cpu().numpy())
                outcome_targets.extend(batch['outcome_label'].cpu().numpy())
                price_preds.extend(outputs['price_prediction'].squeeze().cpu().numpy())
                price_targets.extend(batch['price_target'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        avg_outcome_loss = total_outcome_loss / num_batches
        avg_price_loss = total_price_loss / num_batches
        
        outcome_accuracy = accuracy_score(outcome_targets, outcome_preds)
        price_mse = mean_squared_error(price_targets, price_preds)
        price_mae = mean_absolute_error(price_targets, price_preds)
        price_r2 = r2_score(price_targets, price_preds)
        
        return {
            'total_loss': avg_loss,
            'outcome_loss': avg_outcome_loss,
            'price_loss': avg_price_loss,
            'outcome_accuracy': outcome_accuracy,
            'price_mse': price_mse,
            'price_mae': price_mae,
            'price_r2': price_r2
        }
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics"""
        
        logger.info(f"\nEpoch {epoch + 1} Results:")
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                   f"Outcome Acc: {train_metrics['outcome_accuracy']:.4f}, "
                   f"Price MAE: {train_metrics['price_mae']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                   f"Outcome Acc: {val_metrics['outcome_accuracy']:.4f}, "
                   f"Price MAE: {val_metrics['price_mae']:.4f}")
        
        if 'price_r2' in val_metrics:
            logger.info(f"Val R²: {val_metrics['price_r2']:.4f}")
    
    def _should_save_checkpoint(self, epoch: int, val_metrics: Dict) -> bool:
        """Determine if should save checkpoint"""
        
        # Always save if no previous best
        if not self.best_metrics:
            return True
        
        # Check if this is a new best
        monitor_metric = self.config.monitor_metric.replace('val_', '')
        current_score = val_metrics.get(monitor_metric, val_metrics['total_loss'])
        best_score = self.best_metrics.get(monitor_metric, float('inf'))
        
        return current_score < best_score
    
    def _save_checkpoint(self, epoch: int, val_metrics: Dict, prefix: str = 'best'):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': val_metrics,
            'config': self.config.to_dict(),
            'global_step': self.global_step
        }
        
        checkpoint_path = self.exp_path / f"{prefix}_model_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Update best metrics
        if prefix == 'best' and val_metrics:
            self.best_metrics = val_metrics.copy()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _finalize_training(self, 
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          test_loader: Optional[DataLoader],
                          start_time: float) -> Dict[str, Any]:
        """Finalize training and return results"""
        
        training_time = time.time() - start_time
        
        # Load best model for final evaluation
        best_checkpoint = self._load_best_checkpoint()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Final evaluation
        final_val_metrics = self._validate_epoch(val_loader)
        
        test_metrics = None
        if test_loader is not None:
            test_metrics = self._validate_epoch(test_loader)
        
        # Save training plots
        self._save_training_plots()
        
        # Generate detailed report
        report = self._generate_detailed_report(
            final_val_metrics, test_metrics, training_time
        )
        
        results = {
            'best_metrics': self.best_metrics,
            'final_val_metrics': final_val_metrics,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'total_epochs': self.current_epoch + 1,
            'experiment_path': str(self.exp_path),
            'report': report
        }
        
        # Save results
        results_path = self.exp_path / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def _load_best_checkpoint(self) -> Optional[Dict]:
        """Load best checkpoint"""
        
        checkpoint_files = list(self.exp_path.glob("best_model_epoch_*.pt"))
        if not checkpoint_files:
            return None
        
        # Get latest best checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            logger.info(f"Loaded best checkpoint: {latest_checkpoint}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Could not load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def _save_training_plots(self):
        """Save training visualization plots"""
        
        if not self.metrics_history:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}')
        
        # Loss plots
        if 'train_total_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['train_total_loss'], label='Train')
            axes[0, 0].plot(self.metrics_history['val_total_loss'], label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
        
        # Outcome accuracy
        if 'train_outcome_accuracy' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['train_outcome_accuracy'], label='Train')
            axes[0, 1].plot(self.metrics_history['val_outcome_accuracy'], label='Val')
            axes[0, 1].set_title('Outcome Accuracy')
            axes[0, 1].legend()
        
        # Price MAE
        if 'train_price_mae' in self.metrics_history:
            axes[0, 2].plot(self.metrics_history['train_price_mae'], label='Train')
            axes[0, 2].plot(self.metrics_history['val_price_mae'], label='Val')
            axes[0, 2].set_title('Price MAE')
            axes[0, 2].legend()
        
        # Individual losses
        if 'train_outcome_loss' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['train_outcome_loss'], label='Train')
            axes[1, 0].plot(self.metrics_history['val_outcome_loss'], label='Val')
            axes[1, 0].set_title('Outcome Loss')
            axes[1, 0].legend()
        
        if 'train_price_loss' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['train_price_loss'], label='Train')
            axes[1, 1].plot(self.metrics_history['val_price_loss'], label='Val')
            axes[1, 1].set_title('Price Loss')
            axes[1, 1].legend()
        
        # R² score if available
        if 'val_price_r2' in self.metrics_history:
            axes[1, 2].plot(self.metrics_history['val_price_r2'])
            axes[1, 2].set_title('Price R² Score')
        
        plt.tight_layout()
        plot_path = self.exp_path / "training_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved: {plot_path}")
    
    def _generate_detailed_report(self, 
                                 val_metrics: Dict,
                                 test_metrics: Optional[Dict],
                                 training_time: float) -> Dict[str, Any]:
        """Generate detailed training report"""
        
        report = {
            'experiment_name': self.experiment_name,
            'training_config': self.config.to_dict(),
            'model_config': self.model.config.to_dict(),
            'training_time_seconds': training_time,
            'total_epochs': self.current_epoch + 1,
            'device': str(self.device),
            'model_size': self.model.get_model_size(),
            'final_metrics': {
                'validation': val_metrics,
                'test': test_metrics
            }
        }
        
        return report
    
    def hyperparameter_optimization(self,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader,
                                   n_trials: int = 50,
                                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Hyperparameter optimization using Optuna
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        
        Returns:
            Dictionary with optimization results
        """
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            # Sample hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
            
            # Create new config
            trial_config = TrainingConfig(
                num_epochs=10,  # Shorter for optimization
                learning_rate=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                early_stopping_patience=5
            )
            
            # Create new model with trial config
            model_config = self.model.config
            model_config.dropout_rate = dropout_rate
            trial_model = PlayAnalysisModel(model_config).to(self.device)
            
            # Create trainer
            trial_trainer = ModelTrainer(
                trial_model,
                trial_config,
                device=self.device,
                experiment_dir=self.experiment_dir / "optuna_trials",
                experiment_name=f"trial_{trial.number}"
            )
            
            # Train
            try:
                results = trial_trainer.train(train_loader, val_loader)
                
                # Return validation loss as objective
                return results['best_metrics'].get('total_loss', float('inf'))
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation loss: {best_value:.4f}")
        
        # Save optimization results
        optim_results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        optim_path = self.exp_path / "hyperparameter_optimization.pickle"
        with open(optim_path, 'wb') as f:
            pickle.dump(optim_results, f)
        
        return optim_results
    
    def evaluate_model(self, 
                      test_loader: DataLoader,
                      return_predictions: bool = False) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions
        
        Returns:
            Dictionary with evaluation results
        """
        
        logger.info("Running comprehensive model evaluation")
        
        self.model.eval()
        
        all_outcome_preds = []
        all_outcome_targets = []
        all_price_preds = []
        all_price_targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    batch['text_embedding'],
                    batch['numerical_features'],
                    batch.get('attention_mask'),
                    return_attention=True
                )
                
                # Store predictions
                all_outcome_preds.extend(outputs['outcome_logits'].argmax(dim=1).cpu().numpy())
                all_outcome_targets.extend(batch['outcome_label'].cpu().numpy())
                all_price_preds.extend(outputs['price_prediction'].squeeze().cpu().numpy())
                all_price_targets.extend(batch['price_target'].cpu().numpy())
                
                if 'attention_weights' in outputs:
                    all_attention_weights.extend(outputs['attention_weights'])
        
        # Calculate comprehensive metrics
        evaluation_results = {
            'outcome_metrics': self._calculate_classification_metrics(
                all_outcome_targets, all_outcome_preds
            ),
            'price_metrics': self._calculate_regression_metrics(
                all_price_targets, all_price_preds
            )
        }
        
        if return_predictions:
            evaluation_results['predictions'] = {
                'outcome_predictions': all_outcome_preds,
                'outcome_targets': all_outcome_targets,
                'price_predictions': all_price_preds,
                'price_targets': all_price_targets
            }
        
        # Save evaluation results
        eval_path = self.exp_path / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        return evaluation_results
    
    def _calculate_classification_metrics(self, targets: List, predictions: List) -> Dict:
        """Calculate classification metrics"""
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(targets, predictions),
            'confusion_matrix': confusion_matrix(targets, predictions).tolist()
        }
    
    def _calculate_regression_metrics(self, targets: List, predictions: List) -> Dict:
        """Calculate regression metrics"""
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        }