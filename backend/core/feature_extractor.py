"""
Feature extraction module for contract clause embeddings.

This module uses sentence-transformers to generate semantic embeddings
for contract clauses, which are then used for risk classification.

Requirements: 3.3
"""

import logging
from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import hashlib

from api.models import Clause

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts feature vectors from contract clauses using sentence embeddings.
    
    Uses the all-MiniLM-L6-v2 model from sentence-transformers, which provides:
    - 384-dimensional embeddings
    - Fast inference
    - Good semantic understanding
    - Small model size (~80MB)
    
    Includes caching for performance optimization.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        cache_size: int = 1000
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_size: Maximum number of embeddings to cache
        """
        self.model_name = model_name
        self.cache_size = cache_size
        
        logger.info(f"Loading sentence-transformers model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Cache for storing computed embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def extract(self, clauses: List[Clause]) -> np.ndarray:
        """
        Generate feature embeddings for a list of clauses.
        
        Args:
            clauses: List of Clause objects
            
        Returns:
            Feature matrix of shape (n_clauses, embedding_dim)
            
        Raises:
            ValueError: If clauses list is empty
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> clauses = [Clause(text="The parties agree...", ...)]
            >>> features = extractor.extract(clauses)
            >>> features.shape
            (1, 384)
        """
        if not clauses:
            raise ValueError("Cannot extract features from empty clause list")
        
        # Extract texts from clauses
        texts = [clause.text for clause in clauses]
        
        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if cache_key in self._embedding_cache:
                # Use cached embedding
                embeddings.append((idx, self._embedding_cache[cache_key]))
                self._cache_hits += 1
            else:
                # Need to compute embedding
                texts_to_encode.append(text)
                indices_to_encode.append(idx)
                self._cache_misses += 1
        
        # Compute embeddings for uncached texts
        if texts_to_encode:
            logger.debug(f"Computing embeddings for {len(texts_to_encode)} clauses")
            new_embeddings = self.model.encode(
                texts_to_encode,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Add to cache and results
            for idx, text, embedding in zip(indices_to_encode, texts_to_encode, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._add_to_cache(cache_key, embedding)
                embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        feature_matrix = np.array([emb for _, emb in embeddings])
        
        logger.debug(
            f"Feature extraction complete. Shape: {feature_matrix.shape}, "
            f"Cache hits: {self._cache_hits}, Cache misses: {self._cache_misses}"
        )
        
        return feature_matrix
    
    def extract_single(self, text: str) -> np.ndarray:
        """
        Generate feature embedding for a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector of shape (embedding_dim,)
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> embedding = extractor.extract_single("The parties agree...")
            >>> embedding.shape
            (384,)
        """
        if not text or not text.strip():
            raise ValueError("Cannot extract features from empty text")
        
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        self._add_to_cache(cache_key, embedding)
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.
        
        Uses MD5 hash of normalized text for efficient lookup.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key string
        """
        # Normalize text: lowercase, strip whitespace
        normalized = ' '.join(text.lower().split())
        # Generate hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: np.ndarray) -> None:
        """
        Add an embedding to the cache.
        
        Implements simple LRU-like eviction when cache is full.
        
        Args:
            key: Cache key
            embedding: Embedding to cache
        """
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entry (first key in dict)
            # Note: In Python 3.7+, dicts maintain insertion order
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            logger.debug(f"Cache full, evicted entry: {oldest_key}")
        
        self._embedding_cache[key] = embedding
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hits, misses, and size
        """
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._embedding_cache),
            'max_size': self.cache_size,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) 
                       if (self._cache_hits + self._cache_misses) > 0 else 0.0
        }
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()
    
    def batch_extract(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings for a list of texts with batching.
        
        Useful for processing large datasets efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            Feature matrix of shape (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Cannot extract features from empty text list")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for batch
            batch_embeddings = []
            texts_to_encode = []
            
            for text in batch:
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    batch_embeddings.append(self._embedding_cache[cache_key])
                    self._cache_hits += 1
                else:
                    texts_to_encode.append(text)
                    self._cache_misses += 1
            
            # Encode uncached texts
            if texts_to_encode:
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 100,
                    batch_size=batch_size
                )
                
                # Add to cache
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self._add_to_cache(cache_key, embedding)
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)


# Singleton instance for reuse across the application
_feature_extractor_instance: Optional[FeatureExtractor] = None


def get_feature_extractor(
    model_name: str = 'all-MiniLM-L6-v2',
    cache_size: int = 1000
) -> FeatureExtractor:
    """
    Get or create a singleton FeatureExtractor instance.
    
    This ensures the model is only loaded once and reused across requests.
    
    Args:
        model_name: Name of the sentence-transformers model
        cache_size: Maximum number of embeddings to cache
        
    Returns:
        FeatureExtractor instance
    """
    global _feature_extractor_instance
    
    if _feature_extractor_instance is None:
        _feature_extractor_instance = FeatureExtractor(
            model_name=model_name,
            cache_size=cache_size
        )
    
    return _feature_extractor_instance


if __name__ == "__main__":
    """
    Example usage and testing of the FeatureExtractor.
    """
    # Create sample clauses
    from api.models import Clause
    
    sample_clauses = [
        Clause(
            document_id="test-doc",
            text="The employee agrees to work 40 hours per week.",
            position=0
        ),
        Clause(
            document_id="test-doc",
            text="The company shall provide health insurance benefits.",
            position=1
        ),
        Clause(
            document_id="test-doc",
            text="Either party may terminate this agreement with 30 days notice.",
            position=2
        )
    ]
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("Extracting features for sample clauses...")
    features = extractor.extract(sample_clauses)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Embedding dimension: {extractor.get_embedding_dimension()}")
    
    # Test caching
    print("\nTesting cache...")
    features2 = extractor.extract(sample_clauses)
    
    cache_stats = extractor.get_cache_stats()
    print(f"Cache statistics: {cache_stats}")
    
    # Verify cached results are identical
    assert np.allclose(features, features2), "Cached embeddings should be identical"
    print("✓ Cache working correctly")
    
    # Test single extraction
    print("\nTesting single extraction...")
    single_embedding = extractor.extract_single("This is a test clause.")
    print(f"Single embedding shape: {single_embedding.shape}")
    
    print("\n✓ All tests passed!")
