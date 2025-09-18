"""
Data Loader for NFL Play Analysis

Efficient data loading and preprocessing for transformer training on free GPU instances.
"""

import random
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Single training sample"""
    text_embedding: torch.Tensor
    numerical_features: torch.Tensor
    outcome_label: int
    price_target: float
    attention_mask: torch.Tensor
    metadata: Dict[str, Any]


class NFLDataset(Dataset):
    """
    NFL play dataset for transformer training
    
    Features:
    - Memory-efficient loading
    - Automatic preprocessing
    - Class balancing support
    - Missing data handling
    """
    
    def __init__(self,
                 samples: List[DataSample],
                 augment_data: bool = False,
                 max_samples: Optional[int] = None):
        
        self.samples = samples[:max_samples] if max_samples else samples
        self.augment_data = augment_data
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Apply augmentation if enabled
        if self.augment_data and random.random() < 0.3:
            sample = self._augment_sample(sample)
        
        return {
            'text_embedding': sample.text_embedding,
            'numerical_features': sample.numerical_features,
            'outcome_label': torch.tensor(sample.outcome_label, dtype=torch.long),
            'price_target': torch.tensor(sample.price_target, dtype=torch.float),
            'attention_mask': sample.attention_mask,
            'metadata': sample.metadata
        }
    
    def _augment_sample(self, sample: DataSample) -> DataSample:
        """Apply data augmentation"""
        
        # Add small noise to numerical features
        noise = torch.randn_like(sample.numerical_features) * 0.01
        augmented_numerical = sample.numerical_features + noise
        
        # Add small noise to embeddings
        embedding_noise = torch.randn_like(sample.text_embedding) * 0.001
        augmented_embedding = sample.text_embedding + embedding_noise
        
        return DataSample(
            text_embedding=augmented_embedding,
            numerical_features=augmented_numerical,
            outcome_label=sample.outcome_label,
            price_target=sample.price_target,
            attention_mask=sample.attention_mask,
            metadata=sample.metadata
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        
        labels = [sample.outcome_label for sample in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Inverse frequency weighting
        weights = len(labels) / (len(unique_labels) * counts)
        
        # Create weight tensor
        weight_dict = {label: weight for label, weight in zip(unique_labels, weights)}
        sample_weights = [weight_dict[label] for label in labels]
        
        return torch.tensor(sample_weights, dtype=torch.float)


class NFLDataLoader:
    """
    Data loader for NFL play analysis with preprocessing and optimization for free GPU instances
    
    Features:
    - Automatic data preprocessing
    - Train/validation/test splitting
    - Class balancing
    - Memory-efficient loading
    - GPU memory optimization
    """
    
    def __init__(self,
                 text_processor,  # PlayTextProcessor instance
                 batch_size: int = 16,
                 validation_split: float = 0.2,
                 test_split: float = 0.1,
                 max_sequence_length: int = 128,
                 balance_classes: bool = True,
                 augment_training: bool = True,
                 random_seed: int = 42):
        
        self.text_processor = text_processor
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.max_sequence_length = max_sequence_length
        self.balance_classes = balance_classes
        self.augment_training = augment_training
        self.random_seed = random_seed
        
        # Preprocessing objects
        self.numerical_scaler = StandardScaler()
        self.outcome_encoder = LabelEncoder()
        
        # Data storage
        self.train_dataset: Optional[NFLDataset] = None
        self.val_dataset: Optional[NFLDataset] = None
        self.test_dataset: Optional[NFLDataset] = None
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def prepare_data_from_files(self, 
                               data_files: List[str],
                               play_text_column: str = 'play_description',
                               outcome_column: str = 'play_outcome',
                               price_target_column: str = 'price_change',
                               numerical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare training data from NFL data files
        
        Args:
            data_files: List of paths to data files (CSV/JSON)
            play_text_column: Column containing play descriptions
            outcome_column: Column containing play outcomes
            price_target_column: Column containing price changes
            numerical_columns: List of numerical feature columns
        
        Returns:
            Dictionary with data statistics
        """
        
        logger.info(f"Loading data from {len(data_files)} files")
        
        # Load and combine data
        all_data = []
        for file_path in data_files:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue
                
                all_data.append(df)
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not all_data:
            raise ValueError("No valid data files loaded")
        
        # Combine datasets
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        
        # Prepare features
        return self.prepare_data_from_dataframe(
            combined_data,
            play_text_column,
            outcome_column, 
            price_target_column,
            numerical_columns
        )
    
    def prepare_data_from_dataframe(self,
                                   df: pd.DataFrame,
                                   play_text_column: str = 'play_description',
                                   outcome_column: str = 'play_outcome',
                                   price_target_column: str = 'price_change',
                                   numerical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare training data from DataFrame
        
        Returns:
            Dictionary with data preparation statistics
        """
        
        logger.info("Preparing training data from DataFrame")
        
        # Handle missing data
        df = df.dropna(subset=[play_text_column, outcome_column, price_target_column])
        
        # Default numerical columns if not provided
        if numerical_columns is None:
            numerical_columns = [
                'quarter', 'time_remaining', 'down', 'distance', 'field_position',
                'score_home', 'score_away', 'timeouts_home', 'timeouts_away'
            ]
        
        # Keep only available numerical columns
        available_numerical = [col for col in numerical_columns if col in df.columns]
        if len(available_numerical) < len(numerical_columns):
            logger.warning(f"Missing numerical columns: {set(numerical_columns) - set(available_numerical)}")
        
        # Fill missing numerical values
        for col in available_numerical:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Process text data
        logger.info("Processing play descriptions...")
        play_texts = df[play_text_column].tolist()
        processed_plays = self.text_processor.process_plays(play_texts)
        
        # Filter out plays with poor confidence
        valid_indices = [
            i for i, play in enumerate(processed_plays) 
            if play.confidence > 0.3 and play.embeddings is not None
        ]
        
        logger.info(f"Filtered to {len(valid_indices)} valid plays (confidence > 0.3)")
        
        # Extract features
        text_embeddings = []
        numerical_features = []
        outcomes = []
        price_targets = []
        
        for idx in valid_indices:
            processed_play = processed_plays[idx]
            row = df.iloc[idx]
            
            # Text embeddings
            text_embeddings.append(processed_play.embeddings)
            
            # Numerical features
            numerical_row = []
            for col in available_numerical:
                numerical_row.append(row[col])
            
            # Add extracted entity features
            numerical_row.extend([
                len(processed_play.entities.players),
                len(processed_play.entities.actions),
                processed_play.entities.yardage or 0,
                processed_play.entities.down or 0,
                processed_play.entities.distance or 0,
                processed_play.entities.score_change or 0,
                processed_play.confidence
            ])
            
            numerical_features.append(numerical_row)
            
            # Targets
            outcomes.append(row[outcome_column])
            price_targets.append(row[price_target_column])
        
        # Convert to arrays
        text_embeddings = torch.stack(text_embeddings)
        numerical_features = np.array(numerical_features, dtype=np.float32)
        
        # Preprocess numerical features
        numerical_features = self.numerical_scaler.fit_transform(numerical_features)
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        
        # Encode outcomes
        outcomes = self.outcome_encoder.fit_transform(outcomes)
        
        # Create samples
        samples = []
        for i in range(len(valid_indices)):
            # Create attention mask (all ones for now, can be enhanced)
            attention_mask = torch.ones(text_embeddings[i].shape[0] if text_embeddings[i].dim() > 1 else 1)
            
            sample = DataSample(
                text_embedding=text_embeddings[i],
                numerical_features=numerical_features[i],
                outcome_label=outcomes[i],
                price_target=price_targets[i],
                attention_mask=attention_mask,
                metadata={'original_index': valid_indices[i]}
            )
            samples.append(sample)
        
        # Split data
        train_samples, temp_samples = train_test_split(
            samples, test_size=self.validation_split + self.test_split, 
            random_state=self.random_seed, stratify=outcomes
        )
        
        if self.test_split > 0:
            val_size = self.validation_split / (self.validation_split + self.test_split)
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=1-val_size, 
                random_state=self.random_seed
            )
        else:
            val_samples = temp_samples
            test_samples = []
        
        # Create datasets
        self.train_dataset = NFLDataset(
            train_samples, augment_data=self.augment_training
        )
        self.val_dataset = NFLDataset(val_samples, augment_data=False)
        
        if test_samples:
            self.test_dataset = NFLDataset(test_samples, augment_data=False)
        
        # Data statistics
        stats = {
            'total_samples': len(samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples) if test_samples else 0,
            'num_classes': len(self.outcome_encoder.classes_),
            'class_names': self.outcome_encoder.classes_.tolist(),
            'numerical_feature_dim': numerical_features.shape[1],
            'text_embedding_dim': text_embeddings.shape[1] if text_embeddings.dim() > 1 else text_embeddings.shape[0],
            'class_distribution': dict(zip(*np.unique(outcomes, return_counts=True)))
        }
        
        logger.info(f"Data preparation complete: {stats}")
        return stats
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Get PyTorch data loaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        
        if self.train_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data_* first.")
        
        # Create samplers for class balancing
        if self.balance_classes:
            class_weights = self.train_dataset.get_class_weights()
            sampler = WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(class_weights),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=2,  # Conservative for free tiers
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        test_loader = None
        if self.test_dataset is not None:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
                drop_last=False
            )
        
        return train_loader, val_loader, test_loader
    
    def save_preprocessors(self, filepath: str):
        """Save preprocessing objects"""
        preprocessors = {
            'numerical_scaler': self.numerical_scaler,
            'outcome_encoder': self.outcome_encoder,
            'config': {
                'batch_size': self.batch_size,
                'max_sequence_length': self.max_sequence_length,
                'balance_classes': self.balance_classes
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: str):
        """Load preprocessing objects"""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.numerical_scaler = preprocessors['numerical_scaler']
        self.outcome_encoder = preprocessors['outcome_encoder']
        
        logger.info(f"Preprocessors loaded from {filepath}")
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        numerical_names = [
            'quarter', 'time_remaining', 'down', 'distance', 'field_position',
            'score_home', 'score_away', 'timeouts_home', 'timeouts_away',
            'num_players', 'num_actions', 'yardage', 'entity_down', 
            'entity_distance', 'score_change', 'confidence'
        ]
        
        return numerical_names
    
    def create_synthetic_data(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Create synthetic data for testing (when real data is not available)
        
        Returns:
            Dictionary with data statistics
        """
        
        logger.info(f"Creating {num_samples} synthetic samples for testing")
        
        # Generate synthetic text embeddings
        embedding_dim = 768
        text_embeddings = torch.randn(num_samples, embedding_dim)
        
        # Generate synthetic numerical features
        numerical_dim = 16
        numerical_features = torch.randn(num_samples, numerical_dim)
        
        # Generate synthetic labels
        num_classes = 8
        outcomes = np.random.randint(0, num_classes, num_samples)
        
        # Generate synthetic price targets
        price_targets = np.random.normal(0, 0.05, num_samples)  # Small price changes
        
        # Create samples
        samples = []
        for i in range(num_samples):
            attention_mask = torch.ones(1)  # Single token mask
            
            sample = DataSample(
                text_embedding=text_embeddings[i],
                numerical_features=numerical_features[i],
                outcome_label=outcomes[i],
                price_target=price_targets[i],
                attention_mask=attention_mask,
                metadata={'synthetic': True, 'index': i}
            )
            samples.append(sample)
        
        # Split data
        train_samples, temp_samples = train_test_split(
            samples, test_size=self.validation_split + self.test_split,
            random_state=self.random_seed
        )
        
        if self.test_split > 0:
            val_size = self.validation_split / (self.validation_split + self.test_split)
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=1-val_size,
                random_state=self.random_seed
            )
        else:
            val_samples = temp_samples
            test_samples = []
        
        # Create datasets
        self.train_dataset = NFLDataset(train_samples, augment_data=self.augment_training)
        self.val_dataset = NFLDataset(val_samples, augment_data=False)
        
        if test_samples:
            self.test_dataset = NFLDataset(test_samples, augment_data=False)
        
        # Initialize preprocessors with dummy data
        self.numerical_scaler.fit(numerical_features.numpy())
        self.outcome_encoder.fit(outcomes)
        
        # Statistics
        stats = {
            'total_samples': num_samples,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples) if test_samples else 0,
            'num_classes': num_classes,
            'numerical_feature_dim': numerical_dim,
            'text_embedding_dim': embedding_dim,
            'synthetic': True
        }
        
        logger.info(f"Synthetic data created: {stats}")
        return stats