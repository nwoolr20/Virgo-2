"""6D coordinate system for conversation memory."""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from typing import Optional, List
import re


class ConversationCoordinateSystem:
    """
    Maps conversation turns into 6D coordinate space:
    [temporal, turn_id, semantic, importance, speaker, sentiment]
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            embedding_model: HuggingFace model name for sentence embeddings
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.pca: Optional[PCA] = None
        self.temporal_origin: Optional[float] = None
        self.max_turn_id: int = 10000  # Normalization constant
        
    def fit_semantic_projection(self, texts: List[str]) -> None:
        """
        Learn PCA projection for semantic dimension.
        
        Args:
            texts: List of conversation texts to fit on
        """
        if len(texts) < 2:
            # Can't fit PCA on single sample
            self.pca = None
            return
            
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        self.pca = PCA(n_components=1)
        self.pca.fit(embeddings)
        
    def compute_importance(self, text: str) -> float:
        """
        Heuristic importance score [0-1] based on:
        - Named entities (capitalized words)
        - Personal pronouns
        - Questions
        - Emotional content
        - Length
        
        Args:
            text: Input text
            
        Returns:
            Importance score in [0, 1]
        """
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        
        if not words:
            return 0.0
        
        # Named entities: capitalized words (except sentence start)
        capitals = sum(1 for i, word in enumerate(words) 
                      if i > 0 and word and word[0].isupper())
        score += min(0.3, capitals * 0.1)
        
        # Personal content markers
        personal_markers = [
            r'\bmy\b', r'\bme\b', r'\bi\b', r"\bi'm\b", 
            r"\bi've\b", r'\bmine\b', r'\bmyself\b'
        ]
        if any(re.search(marker, text_lower) for marker in personal_markers):
            score += 0.2
            
        # Questions
        if '?' in text:
            score += 0.2
            
        # Emotional words
        emotional_words = [
            'feel', 'love', 'hate', 'sad', 'happy', 'angry', 
            'afraid', 'worried', 'excited', 'anxious', 'grateful'
        ]
        if any(word in text_lower for word in emotional_words):
            score += 0.2
            
        # Length (prefer substantial content, 20 words = 0.1)
        word_count = len(words)
        score += min(0.1, word_count / 200)
        
        return min(1.0, score)
    
    def compute_sentiment(self, text: str) -> float:
        """
        Compute sentiment polarity, mapped to [0, 1].
        
        Args:
            text: Input text
            
        Returns:
            Sentiment in [0, 1] where 0.5 is neutral
        """
        try:
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity
            # Map [-1, 1] to [0, 1]
            return (polarity + 1.0) / 2.0
        except Exception:
            # Fallback to neutral if TextBlob unavailable
            return 0.5
    
    def extract_coordinates(
        self,
        text: str,
        timestamp: float,
        turn_id: int,
        speaker_id: int
    ) -> torch.Tensor:
        """
        Convert conversation turn to 6D coordinates.
        
        Args:
            text: Conversation text
            timestamp: Unix timestamp
            turn_id: Sequential turn number
            speaker_id: 0 for user, 1 for assistant
            
        Returns:
            6D coordinate tensor in [0, 1]^6
        """
        # Initialize temporal origin on first call
        if self.temporal_origin is None:
            self.temporal_origin = timestamp
            
        # Dimension 0: Temporal (days since start)
        temporal_days = (timestamp - self.temporal_origin) / (24 * 3600)
        # Sigmoid normalization for unbounded time
        temporal = 1.0 / (1.0 + np.exp(-temporal_days / 10))
        
        # Dimension 1: Turn ID (normalized)
        turn_normalized = min(1.0, turn_id / self.max_turn_id)
        
        # Dimension 2: Semantic projection
        if self.pca is None:
            semantic = 0.5  # Default to center
        else:
            try:
                embedding = self.encoder.encode([text], convert_to_numpy=True)[0]
                semantic_raw = self.pca.transform([embedding])[0][0]
                # Sigmoid normalization
                semantic = 1.0 / (1.0 + np.exp(-semantic_raw / 10))
            except Exception:
                semantic = 0.5
                
        # Dimension 3: Importance
        importance = self.compute_importance(text)
        
        # Dimension 4: Speaker ID (already binary)
        speaker = float(speaker_id)
        
        # Dimension 5: Sentiment
        sentiment = self.compute_sentiment(text)
        
        coords = torch.tensor([
            temporal,
            turn_normalized,
            semantic,
            importance,
            speaker,
            sentiment
        ], dtype=torch.float32)
        
        return coords
