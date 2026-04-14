"""
RAG (Retrieval-Augmented Generation) System for legal guideline retrieval.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import LegalGuideline - handle both direct and module imports
try:
    from backend.api.models import LegalGuideline
except ImportError:
    from api.models import LegalGuideline


logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG system for retrieving relevant legal guidelines.
    
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Chunks documents into 500-token segments with 50-token overlap.
    """
    
    def __init__(self, vector_store_path: str = "backend/data/vector_store"):
        """
        Initialize RAG system.
        
        Args:
            vector_store_path: Path to persist ChromaDB vector store
        """
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client with telemetry disabled for compatibility
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("legal_guidelines")
            logger.info("Loaded existing legal_guidelines collection")
        except Exception:
            self.collection = self.client.create_collection(
                name="legal_guidelines",
                metadata={"description": "Legal guidelines and contract law documents"}
            )
            logger.info("Created new legal_guidelines collection")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into segments with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Overlap size in tokens (approximate)
            
        Returns:
            List of text chunks
        """
        # Simple word-based chunking (approximation of tokens)
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            # Move start forward by (chunk_size - overlap)
            start += (chunk_size - overlap)
            
            # Break if we've covered all words
            if end >= len(words):
                break
        
        return chunks
    
    def index_documents(self, documents: List[dict]) -> None:
        """
        Index legal documents in vector store.
        
        Args:
            documents: List of document dicts with keys:
                - text: Document text content
                - source: Source name/title
                - url: Optional URL
        """
        logger.info(f"Indexing {len(documents)} documents")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            source = doc.get("source", f"Document {doc_idx}")
            url = doc.get("url", "")
            
            # Chunk the document
            chunks = self.chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": source,
                    "url": url,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx
                })
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
                chunk_id += 1
        
        if not all_chunks:
            logger.warning("No chunks to index")
            return
        
        # Generate embeddings and add to collection
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        logger.info(f"Successfully indexed {len(all_chunks)} chunks from {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[LegalGuideline]:
        """
        Retrieve relevant legal guidelines for a query.
        
        Args:
            query: Risk description or clause text
            top_k: Number of guidelines to retrieve (default 3)
            
        Returns:
            List of LegalGuideline objects with source citations
        """
        # Check if collection is empty
        count = self.collection.count()
        if count == 0:
            logger.warning("Vector store is empty, no guidelines to retrieve")
            return []
        
        # Limit top_k to available documents
        top_k = min(top_k, count)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results as LegalGuideline objects
        guidelines = []
        
        if results and results['documents'] and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to relevance score (0-1, higher is better)
                # ChromaDB uses L2 distance, so smaller is better
                # We'll use a simple exponential decay: exp(-distance)
                import math
                relevance_score = math.exp(-distance)
                
                guideline = LegalGuideline(
                    text=doc,
                    source=metadata.get("source", "Unknown"),
                    url=metadata.get("url") if metadata.get("url") else None,
                    relevance_score=relevance_score
                )
                guidelines.append(guideline)
        
        logger.info(f"Retrieved {len(guidelines)} guidelines for query")
        return guidelines
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection("legal_guidelines")
            self.collection = self.client.create_collection(
                name="legal_guidelines",
                metadata={"description": "Legal guidelines and contract law documents"}
            )
            logger.info("Cleared legal_guidelines collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }
