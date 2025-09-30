"""Complete memory storage and retrieval system."""

import torch
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from .coordinates import ConversationCoordinateSystem
from .field import ConversationField


@dataclass
class Memory:
    """Single conversation memory."""
    text: str
    coordinates: torch.Tensor
    embedding: torch.Tensor
    timestamp: float
    turn_id: int
    speaker_id: int
    
    def to_dict(self):
        """Convert to serializable dict."""
        return {
            'text': self.text,
            'coordinates': self.coordinates.tolist(),
            'embedding': self.embedding.tolist(),
            'timestamp': self.timestamp,
            'turn_id': self.turn_id,
            'speaker_id': self.speaker_id
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        """Load from dict."""
        return cls(
            text=d['text'],
            coordinates=torch.tensor(d['coordinates']),
            embedding=torch.tensor(d['embedding']),
            timestamp=d['timestamp'],
            turn_id=d['turn_id'],
            speaker_id=d['speaker_id']
        )


class MemorySystem:
    """
    Complete memory system with neural field + FAISS retrieval.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        coord_system: Optional[ConversationCoordinateSystem] = None,
        field_config: Optional[dict] = None
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            coord_system: Custom coordinate system (optional)
            field_config: Field architecture config (optional)
        """
        self.embedding_dim = embedding_dim
        self.coord_system = coord_system or ConversationCoordinateSystem()
        
        # Create field
        field_config = field_config or {}
        self.field = ConversationField(
            embedding_dim=embedding_dim,
            **field_config
        )
        
        # FAISS index for fast retrieval
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Memory storage
        self.memories: List[Memory] = []
        self.is_fitted = False
        
    def store(self, text: str, speaker_id: int, timestamp: Optional[float] = None):
        """
        Store new conversation turn.
        
        Args:
            text: Conversation text
            speaker_id: 0=user, 1=assistant
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        turn_id = len(self.memories)
        
        # Extract coordinates
        coords = self.coord_system.extract_coordinates(
            text, timestamp, turn_id, speaker_id
        )
        
        # Get embedding
        embedding = torch.tensor(
            self.coord_system.encoder.encode([text])[0],
            dtype=torch.float32
        )
        
        # Create memory
        memory = Memory(
            text=text,
            coordinates=coords,
            embedding=embedding,
            timestamp=timestamp,
            turn_id=turn_id,
            speaker_id=speaker_id
        )
        
        self.memories.append(memory)
        
        # Add to FAISS
        self.index.add(embedding.unsqueeze(0).numpy())
        
        # Mark as needing retraining
        self.is_fitted = False
        
    def fit_field(
        self,
        num_steps: int = 5000,
        lr: float = 1e-4,
        verbose: bool = True
    ) -> dict:
        """
        Train neural field on all stored memories.
        
        Args:
            num_steps: Training iterations
            lr: Learning rate
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        if len(self.memories) == 0:
            raise ValueError("No memories to train on")
            
        # Fit semantic projection if needed
        if self.coord_system.pca is None and len(self.memories) >= 2:
            texts = [m.text for m in self.memories]
            self.coord_system.fit_semantic_projection(texts)
            
            # Recompute coordinates with fitted PCA
            for memory in self.memories:
                memory.coordinates = self.coord_system.extract_coordinates(
                    memory.text,
                    memory.timestamp,
                    memory.turn_id,
                    memory.speaker_id
                )
        
        # Prepare training batch
        coords = torch.stack([m.coordinates for m in self.memories])
        embeddings = torch.stack([m.embedding for m in self.memories])
        
        # Train field
        if verbose:
            print(f"Training field on {len(self.memories)} memories...")
            
        metrics = self.field.fit_memory(coords, embeddings, num_steps, lr, verbose)
        self.is_fitted = True
        
        return metrics
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Memory, float]]:
        """
        Retrieve k most relevant memories.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (memory, distance) tuples
        """
        # Encode query
        query_embedding = self.coord_system.encoder.encode([query])[0]
        query_tensor = torch.tensor(query_embedding).unsqueeze(0).numpy()
        
        # Search FAISS index
        k_actual = min(k, len(self.memories))
        distances, indices = self.index.search(query_tensor, k_actual)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.memories):
                results.append((self.memories[idx], float(dist)))
        
        return results
    
    def query_field(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Query field at arbitrary coordinates (for interpolation).
        
        Args:
            coordinates: [N, 6] coordinate tensor
            
        Returns:
            Predicted embeddings [N, embedding_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("Field not fitted. Call fit_field() first.")
            
        with torch.no_grad():
            return self.field(coordinates)
    
    def save(self, path: Path):
        """
        Save entire system to disk.
        
        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save memories
        memories_data = [m.to_dict() for m in self.memories]
        with open(path / "memories.json", 'w') as f:
            json.dump(memories_data, f)
        
        # Save field weights
        torch.save(self.field.state_dict(), path / "field.pt")
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save coordinate system state
        coord_state = {
            "temporal_origin": self.coord_system.temporal_origin,
            "max_turn_id": self.coord_system.max_turn_id,
            "has_pca": self.coord_system.pca is not None
        }
        with open(path / "coord_state.json", 'w') as f:
            json.dump(coord_state, f)
        
        # Save PCA if it exists
        if self.coord_system.pca is not None:
            import joblib
            joblib.dump(self.coord_system.pca, path / "pca.pkl")
        
        # Save metadata
        metadata = {
            "is_fitted": self.is_fitted,
            "num_memories": len(self.memories),
            "embedding_dim": self.embedding_dim
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load(self, path: Path):
        """
        Load entire system from disk.
        
        Args:
            path: Directory to load from
        """
        path = Path(path)
        
        # Load memories
        with open(path / "memories.json", 'r') as f:
            memories_data = json.load(f)
        self.memories = [Memory.from_dict(m) for m in memories_data]
        
        # Load field weights
        self.field.load_state_dict(torch.load(path / "field.pt"))
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load coordinate system state
        with open(path / "coord_state.json", 'r') as f:
            coord_state = json.load(f)
        
        self.coord_system.temporal_origin = coord_state["temporal_origin"]
        self.coord_system.max_turn_id = coord_state["max_turn_id"]
        
        # Load PCA if it exists
        if coord_state["has_pca"]:
            import joblib
            self.coord_system.pca = joblib.load(path / "pca.pkl")
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        self.is_fitted = metadata["is_fitted"]
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "total_memories": len(self.memories),
            "user_turns": sum(1 for m in self.memories if m.speaker_id == 0),
            "assistant_turns": sum(1 for m in self.memories if m.speaker_id == 1),
            "is_fitted": self.is_fitted,
            "has_pca": self.coord_system.pca is not None
        }
