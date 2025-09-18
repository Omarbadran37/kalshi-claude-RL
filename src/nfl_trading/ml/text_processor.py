"""
Play Text Processor

Comprehensive text processing pipeline for NFL play descriptions with entity extraction,
normalization, and embedding generation optimized for free GPU instances.
"""

import re
import string
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class PlayEntities:
    """Extracted entities from a play description"""
    players: List[str]
    actions: List[str]
    outcomes: List[str]
    positions: List[str]
    yardage: Optional[int]
    down: Optional[int]
    distance: Optional[int]
    field_position: Optional[str]
    time_remaining: Optional[str]
    score_change: Optional[int]


@dataclass
class ProcessedPlay:
    """Processed play with all extracted information"""
    original_text: str
    cleaned_text: str
    normalized_text: str
    entities: PlayEntities
    embeddings: Optional[torch.Tensor]
    confidence: float
    metadata: Dict[str, Any]


class PlayTextProcessor:
    """
    Comprehensive text processing pipeline for NFL play descriptions.
    
    Features:
    - Text cleaning and normalization
    - NFL-specific entity extraction
    - Embedding generation with small transformers
    - Memory-efficient processing for free GPU tiers
    - Robust error handling for corrupted data
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",  # Small, efficient model
        max_length: int = 128,
        batch_size: int = 16,  # Conservative for free tiers
        device: str = "auto",
        cache_embeddings: bool = True
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models with error handling
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = None
        
        # NFL-specific patterns and vocabularies
        self._init_nfl_patterns()
        
        # Caching
        self.embedding_cache = {} if cache_embeddings else None
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load tokenizer and model with memory optimization"""
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                truncation_side='left'
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimization
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map=self.device if self.device.type == 'cuda' else None
            )
            
            if self.device.type == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Initialize TF-IDF for backup embeddings
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to TF-IDF only
            self.tokenizer = None
            self.model = None
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            logger.warning("Falling back to TF-IDF embeddings only")
    
    def _init_nfl_patterns(self):
        """Initialize NFL-specific patterns and vocabularies"""
        
        # Player name patterns
        self.player_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z]\.\s*[A-Z][a-z]+\b',      # F. Last
            r'\b[A-Z][a-z]+\b(?=\s+(?:pass|rush|kick|punt|fumble|intercept))',  # Single name before action
        ]
        
        # Action keywords
        self.action_keywords = {
            'pass': ['pass', 'threw', 'completion', 'incomplete', 'sack', 'interception'],
            'rush': ['rush', 'run', 'carry', 'scramble'],
            'kick': ['field goal', 'extra point', 'punt', 'kickoff'],
            'turnover': ['fumble', 'interception', 'turnover'],
            'penalty': ['penalty', 'flag', 'false start', 'holding', 'offside'],
            'score': ['touchdown', 'field goal', 'safety', 'score']
        }
        
        # Outcome patterns
        self.outcome_patterns = {
            'success': ['touchdown', 'first down', 'completion', 'good'],
            'failure': ['incomplete', 'sack', 'fumble', 'interception', 'missed'],
            'neutral': ['punt', 'timeout', 'spike']
        }
        
        # Yardage pattern
        self.yardage_pattern = r'(?:for\s+)?(-?\d+)\s*yard[s]?'
        
        # Down and distance patterns
        self.down_pattern = r'(\d+)(?:st|nd|rd|th)\s+(?:and|&)\s+(\d+)'
        self.down_simple_pattern = r'(\d+)(?:st|nd|rd|th)\s+down'
        
        # Field position pattern
        self.field_position_pattern = r'\b([A-Z]{2,3})\s+(\d+)\b'
        
        # Time pattern
        self.time_pattern = r'(\d+):(\d+)'
        
        # Score change pattern
        self.score_pattern = r'(?:touchdown|field goal|safety|score)'
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize play description text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\-\.\,\:\(\)]', '', text)
        
        # Normalize common abbreviations
        text = re.sub(r'\byd[s]?\b', 'yard', text, flags=re.IGNORECASE)
        text = re.sub(r'\btd\b', 'touchdown', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfg\b', 'field goal', text, flags=re.IGNORECASE)
        text = re.sub(r'\bint\b', 'interception', text, flags=re.IGNORECASE)
        
        # Standardize down notation
        text = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', text)
        
        return text.strip()
    
    def extract_entities(self, text: str) -> PlayEntities:
        """Extract NFL-specific entities from play description"""
        
        # Initialize entity storage
        players = []
        actions = []
        outcomes = []
        positions = []
        yardage = None
        down = None
        distance = None
        field_position = None
        time_remaining = None
        score_change = None
        
        text_lower = text.lower()
        
        # Extract players
        for pattern in self.player_patterns:
            matches = re.findall(pattern, text)
            players.extend([match.strip() for match in matches])
        
        # Remove duplicates while preserving order
        players = list(dict.fromkeys(players))
        
        # Extract actions
        for action_type, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    actions.append(action_type)
                    break
        
        # Extract outcomes
        for outcome_type, keywords in self.outcome_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    outcomes.append(outcome_type)
                    break
        
        # Extract yardage
        yardage_match = re.search(self.yardage_pattern, text_lower)
        if yardage_match:
            try:
                yardage = int(yardage_match.group(1))
            except ValueError:
                pass
        
        # Extract down and distance
        down_match = re.search(self.down_pattern, text_lower)
        if down_match:
            try:
                down = int(down_match.group(1))
                distance = int(down_match.group(2))
            except ValueError:
                pass
        else:
            # Try simple down pattern
            down_simple_match = re.search(self.down_simple_pattern, text_lower)
            if down_simple_match:
                try:
                    down = int(down_simple_match.group(1))
                except ValueError:
                    pass
        
        # Extract field position
        field_match = re.search(self.field_position_pattern, text.upper())
        if field_match:
            field_position = f"{field_match.group(1)} {field_match.group(2)}"
        
        # Extract time
        time_match = re.search(self.time_pattern, text)
        if time_match:
            time_remaining = f"{time_match.group(1)}:{time_match.group(2)}"
        
        # Detect score change
        if any(keyword in text_lower for keyword in ['touchdown', 'field goal', 'safety']):
            if 'touchdown' in text_lower:
                score_change = 6
            elif 'field goal' in text_lower:
                score_change = 3
            elif 'safety' in text_lower:
                score_change = 2
        
        return PlayEntities(
            players=players,
            actions=list(set(actions)),
            outcomes=list(set(outcomes)),
            positions=positions,
            yardage=yardage,
            down=down,
            distance=distance,
            field_position=field_position,
            time_remaining=time_remaining,
            score_change=score_change
        )
    
    def generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for list of texts with memory optimization"""
        
        if not texts:
            return torch.empty(0, 768)  # Default embedding size
        
        # Check cache first
        if self.embedding_cache is not None:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    cached_embeddings.append((i, self.embedding_cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_embeddings = []
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        
        if uncached_texts and self.model is not None and self.tokenizer is not None:
            try:
                # Process in batches to manage memory
                for i in range(0, len(uncached_texts), self.batch_size):
                    batch_texts = uncached_texts[i:i + self.batch_size]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                            outputs = self.model(**inputs)
                            # Use mean pooling over sequence length
                            embeddings = outputs.last_hidden_state.mean(dim=1)
                            embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    new_embeddings.append(embeddings.cpu())
                    
                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if new_embeddings:
                    new_embeddings = torch.cat(new_embeddings, dim=0)
                
            except Exception as e:
                logger.warning(f"Error generating transformer embeddings: {e}")
                new_embeddings = self._fallback_embeddings(uncached_texts)
        
        else:
            # Fallback to TF-IDF embeddings
            new_embeddings = self._fallback_embeddings(uncached_texts)
        
        # Cache new embeddings
        if self.embedding_cache is not None and len(uncached_texts) > 0:
            for text, embedding in zip(uncached_texts, new_embeddings):
                self.embedding_cache[text] = embedding
        
        # Combine cached and new embeddings
        if cached_embeddings:
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for idx, embedding in cached_embeddings:
                all_embeddings[idx] = embedding
            
            # Place new embeddings
            for i, idx in enumerate(uncached_indices):
                if i < len(new_embeddings):
                    all_embeddings[idx] = new_embeddings[i]
            
            # Convert to tensor
            all_embeddings = torch.stack([emb for emb in all_embeddings if emb is not None])
        else:
            all_embeddings = new_embeddings
        
        return all_embeddings
    
    def _fallback_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate TF-IDF embeddings as fallback"""
        try:
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or len(texts) == 0:
                # Fit on current texts if not fitted
                self.tfidf_vectorizer.fit(texts if texts else ["dummy text"])
            
            embeddings = self.tfidf_vectorizer.transform(texts).toarray()
            return torch.tensor(embeddings, dtype=torch.float32)
        
        except Exception as e:
            logger.warning(f"Error generating TF-IDF embeddings: {e}")
            # Return random embeddings as last resort
            return torch.randn(len(texts), 100)
    
    def process_play(self, text: str, metadata: Optional[Dict] = None) -> ProcessedPlay:
        """Process a single play description"""
        
        metadata = metadata or {}
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return ProcessedPlay(
                original_text=text,
                cleaned_text="",
                normalized_text="",
                entities=PlayEntities([], [], [], [], None, None, None, None, None, None),
                embeddings=None,
                confidence=0.0,
                metadata=metadata
            )
        
        # Extract entities
        entities = self.extract_entities(cleaned_text)
        
        # Generate normalized text
        normalized_text = self._normalize_text(cleaned_text, entities)
        
        # Generate embeddings
        try:
            embeddings = self.generate_embeddings([normalized_text])
            if len(embeddings) > 0:
                embeddings = embeddings[0]
            else:
                embeddings = None
        except Exception as e:
            logger.warning(f"Error generating embeddings for play: {e}")
            embeddings = None
        
        # Calculate confidence score
        confidence = self._calculate_confidence(cleaned_text, entities)
        
        return ProcessedPlay(
            original_text=text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            entities=entities,
            embeddings=embeddings,
            confidence=confidence,
            metadata=metadata
        )
    
    def process_plays(self, plays: List[str], metadata: Optional[List[Dict]] = None) -> List[ProcessedPlay]:
        """Process multiple play descriptions efficiently"""
        
        if metadata is None:
            metadata = [{}] * len(plays)
        
        processed_plays = []
        
        # Clean all texts first
        cleaned_texts = [self.clean_text(play) for play in plays]
        
        # Extract entities for all plays
        all_entities = [self.extract_entities(text) for text in cleaned_texts]
        
        # Generate normalized texts
        normalized_texts = [
            self._normalize_text(cleaned, entities) 
            for cleaned, entities in zip(cleaned_texts, all_entities)
        ]
        
        # Generate embeddings in batch
        try:
            valid_texts = [text for text in normalized_texts if text]
            if valid_texts:
                all_embeddings = self.generate_embeddings(valid_texts)
            else:
                all_embeddings = torch.empty(0, 768)
        except Exception as e:
            logger.warning(f"Error generating batch embeddings: {e}")
            all_embeddings = torch.empty(0, 768)
        
        # Create processed plays
        embedding_idx = 0
        for i, (original, cleaned, normalized, entities, meta) in enumerate(
            zip(plays, cleaned_texts, normalized_texts, all_entities, metadata)
        ):
            # Get embedding if available
            if normalized and embedding_idx < len(all_embeddings):
                embeddings = all_embeddings[embedding_idx]
                embedding_idx += 1
            else:
                embeddings = None
            
            # Calculate confidence
            confidence = self._calculate_confidence(cleaned, entities)
            
            processed_play = ProcessedPlay(
                original_text=original,
                cleaned_text=cleaned,
                normalized_text=normalized,
                entities=entities,
                embeddings=embeddings,
                confidence=confidence,
                metadata=meta
            )
            
            processed_plays.append(processed_play)
        
        return processed_plays
    
    def _normalize_text(self, text: str, entities: PlayEntities) -> str:
        """Create normalized representation of play"""
        
        # Start with cleaned text
        normalized = text.lower()
        
        # Replace player names with generic tokens
        for i, player in enumerate(entities.players):
            normalized = normalized.replace(player.lower(), f"<player_{i}>")
        
        # Add entity information
        entity_tokens = []
        
        if entities.down is not None:
            entity_tokens.append(f"<down_{entities.down}>")
        
        if entities.distance is not None:
            entity_tokens.append(f"<distance_{entities.distance}>")
        
        if entities.yardage is not None:
            entity_tokens.append(f"<yards_{entities.yardage}>")
        
        for action in entities.actions:
            entity_tokens.append(f"<action_{action}>")
        
        for outcome in entities.outcomes:
            entity_tokens.append(f"<outcome_{outcome}>")
        
        if entity_tokens:
            normalized = normalized + " " + " ".join(entity_tokens)
        
        return normalized.strip()
    
    def _calculate_confidence(self, text: str, entities: PlayEntities) -> float:
        """Calculate confidence score for processed play"""
        
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Boost for extracted entities
        if entities.players:
            score += 0.1 * min(len(entities.players), 3) / 3
        
        if entities.actions:
            score += 0.15
        
        if entities.yardage is not None:
            score += 0.1
        
        if entities.down is not None:
            score += 0.05
        
        if entities.outcomes:
            score += 0.1
        
        # Boost for text quality
        word_count = len(text.split())
        if word_count >= 5:
            score += 0.1
        
        # Penalty for very short text
        if word_count < 3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def get_feature_vector(self, processed_play: ProcessedPlay, include_embeddings: bool = True) -> np.ndarray:
        """Create feature vector from processed play"""
        
        features = []
        
        # Entity-based features
        features.extend([
            len(processed_play.entities.players),
            len(processed_play.entities.actions),
            len(processed_play.entities.outcomes),
            processed_play.entities.yardage or 0,
            processed_play.entities.down or 0,
            processed_play.entities.distance or 0,
            processed_play.entities.score_change or 0,
            processed_play.confidence
        ])
        
        # Text length features
        features.extend([
            len(processed_play.original_text),
            len(processed_play.cleaned_text.split()),
            len(processed_play.normalized_text.split())
        ])
        
        # Action type one-hot encoding
        action_types = ['pass', 'rush', 'kick', 'turnover', 'penalty', 'score']
        for action_type in action_types:
            features.append(1.0 if action_type in processed_play.entities.actions else 0.0)
        
        # Outcome type one-hot encoding
        outcome_types = ['success', 'failure', 'neutral']
        for outcome_type in outcome_types:
            features.append(1.0 if outcome_type in processed_play.entities.outcomes else 0.0)
        
        # Include embeddings if available and requested
        if include_embeddings and processed_play.embeddings is not None:
            embedding_features = processed_play.embeddings.numpy()
            features.extend(embedding_features)
        
        return np.array(features, dtype=np.float32)
    
    def similarity_search(self, query_play: ProcessedPlay, candidate_plays: List[ProcessedPlay], top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar plays using embeddings"""
        
        if query_play.embeddings is None:
            return []
        
        similarities = []
        query_embedding = query_play.embeddings.unsqueeze(0)
        
        for i, candidate in enumerate(candidate_plays):
            if candidate.embeddings is not None:
                candidate_embedding = candidate.embeddings.unsqueeze(0)
                similarity = F.cosine_similarity(query_embedding, candidate_embedding).item()
                similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_cache(self, filepath: str):
        """Save embedding cache to disk"""
        if self.embedding_cache is not None:
            torch.save(self.embedding_cache, filepath)
            logger.info(f"Saved embedding cache to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load embedding cache from disk"""
        try:
            self.embedding_cache = torch.load(filepath, map_location='cpu')
            logger.info(f"Loaded embedding cache from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load cache from {filepath}: {e}")
            self.embedding_cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'cache_size': len(self.embedding_cache) if self.embedding_cache else 0,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None
        }