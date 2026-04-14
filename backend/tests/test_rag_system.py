"""
Unit tests for RAG system.
"""

import os
import pytest
import sys
from pathlib import Path

# Set environment variables before importing anything
os.environ.setdefault('GEMINI_API_KEY', 'test_key_for_testing')
os.environ.setdefault('LLM_PROVIDER', 'gemini')

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.rag_system import RAGSystem
from api.models import LegalGuideline


class TestRAGSystem:
    """Test cases for RAG system."""
    
    @pytest.fixture
    def rag_system(self, tmp_path):
        """Create a RAG system with temporary vector store."""
        return RAGSystem(vector_store_path=str(tmp_path / "test_vector_store"))
    
    def test_initialization(self, rag_system):
        """Test RAG system initializes correctly."""
        assert rag_system is not None
        assert rag_system.embedding_model is not None
        assert rag_system.collection is not None
    
    def test_chunk_text_small(self, rag_system):
        """Test chunking small text returns single chunk."""
        text = "This is a small text."
        chunks = rag_system.chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_large(self, rag_system):
        """Test chunking large text creates multiple chunks with overlap."""
        # Create text with 1000 words
        words = ["word"] * 1000
        text = " ".join(words)
        
        chunks = rag_system.chunk_text(text, chunk_size=500, overlap=50)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be approximately 500 words
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            chunk_words = chunk.split()
            assert len(chunk_words) <= 500
    
    def test_index_and_retrieve(self, rag_system):
        """Test indexing documents and retrieving relevant ones."""
        # Index sample documents
        documents = [
            {
                "text": "Liability clauses limit financial exposure in contracts. "
                        "Unlimited liability is risky and should be avoided.",
                "source": "Contract Law Guide",
                "url": "https://example.com/liability"
            },
            {
                "text": "Confidentiality agreements protect sensitive information. "
                        "Non-disclosure clauses are essential for business contracts.",
                "source": "NDA Guidelines",
                "url": "https://example.com/nda"
            }
        ]
        
        rag_system.index_documents(documents)
        
        # Retrieve documents about liability
        results = rag_system.retrieve("What are liability clauses?", top_k=2)
        
        # Should return results
        assert len(results) > 0
        assert len(results) <= 2
        
        # Results should be LegalGuideline objects
        assert all(isinstance(r, LegalGuideline) for r in results) or \
               all(type(r).__name__ == 'LegalGuideline' for r in results)
        
        # First result should be about liability (more relevant)
        assert "liability" in results[0].text.lower()
        
        # Check relevance scores
        assert all(0 <= r.relevance_score <= 1 for r in results)
    
    def test_retrieve_empty_store(self, rag_system):
        """Test retrieving from empty vector store returns empty list."""
        results = rag_system.retrieve("test query", top_k=3)
        assert results == []
    
    def test_retrieve_top_k_constraint(self, rag_system):
        """Test that retrieve respects top_k parameter."""
        # Index 5 documents
        documents = [
            {"text": f"Document {i} about contracts", "source": f"Source {i}", "url": ""}
            for i in range(5)
        ]
        rag_system.index_documents(documents)
        
        # Request top 3
        results = rag_system.retrieve("contracts", top_k=3)
        
        # Should return exactly 3 (or fewer if less available)
        assert len(results) <= 3
    
    def test_get_collection_stats(self, rag_system):
        """Test getting collection statistics."""
        # Initially empty
        stats = rag_system.get_collection_stats()
        assert stats["total_chunks"] == 0
        assert stats["collection_name"] == "legal_guidelines"
        
        # After indexing
        documents = [
            {"text": "Test document", "source": "Test", "url": ""}
        ]
        rag_system.index_documents(documents)
        
        stats = rag_system.get_collection_stats()
        assert stats["total_chunks"] == 1
    
    def test_clear_collection(self, rag_system):
        """Test clearing collection removes all documents."""
        # Index documents
        documents = [
            {"text": "Test document", "source": "Test", "url": ""}
        ]
        rag_system.index_documents(documents)
        
        # Verify indexed
        assert rag_system.get_collection_stats()["total_chunks"] == 1
        
        # Clear
        rag_system.clear_collection()
        
        # Verify empty
        assert rag_system.get_collection_stats()["total_chunks"] == 0
    
    def test_index_empty_documents(self, rag_system):
        """Test indexing empty document list."""
        rag_system.index_documents([])
        stats = rag_system.get_collection_stats()
        assert stats["total_chunks"] == 0
    
    def test_relevance_score_ordering(self, rag_system):
        """Test that results are ordered by relevance."""
        documents = [
            {
                "text": "Liability clauses are very important in contracts.",
                "source": "Source 1",
                "url": ""
            },
            {
                "text": "Confidentiality is also important.",
                "source": "Source 2",
                "url": ""
            },
            {
                "text": "Liability and indemnification protect parties.",
                "source": "Source 3",
                "url": ""
            }
        ]
        rag_system.index_documents(documents)
        
        results = rag_system.retrieve("liability clauses", top_k=3)
        
        # Results should be ordered by relevance (descending)
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score
