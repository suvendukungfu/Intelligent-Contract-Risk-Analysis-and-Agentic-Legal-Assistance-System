#!/usr/bin/env python3
"""
Script to build and populate the RAG vector store with legal documents.

This script:
1. Loads all legal documents from backend/data/legal_documents/
2. Indexes them in ChromaDB vector store
3. Saves the vector store to backend/data/vector_store/

Usage:
    python scripts/setup_rag.py
"""

import os
import sys
from pathlib import Path
import logging

# Set dummy environment variables to avoid config validation errors
os.environ.setdefault('GEMINI_API_KEY', 'dummy_key_for_setup')
os.environ.setdefault('LLM_PROVIDER', 'gemini')

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from core.rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_legal_documents(documents_dir: str = "backend/data/legal_documents") -> list:
    """
    Load all legal documents from the specified directory.
    
    Args:
        documents_dir: Path to directory containing legal documents
        
    Returns:
        List of document dicts with text, source, and url
    """
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        logger.error(f"Documents directory not found: {documents_path}")
        return []
    
    documents = []
    
    # Load all .txt files from the directory
    for file_path in documents_path.glob("*.txt"):
        logger.info(f"Loading document: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create document dict
            doc = {
                "text": text,
                "source": file_path.stem.replace('_', ' ').title(),
                "url": ""  # Could be populated if documents have associated URLs
            }
            documents.append(doc)
            
            logger.info(f"Loaded {len(text)} characters from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def main():
    """Main function to set up RAG system."""
    logger.info("Starting RAG system setup")
    
    # Load legal documents
    logger.info("Loading legal documents...")
    documents = load_legal_documents()
    
    if not documents:
        logger.error("No documents found to index. Please add legal documents to backend/data/legal_documents/")
        sys.exit(1)
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem(vector_store_path="backend/data/vector_store")
    
    # Clear existing collection (optional - comment out to append)
    logger.info("Clearing existing collection...")
    rag_system.clear_collection()
    
    # Index documents
    logger.info("Indexing documents in vector store...")
    rag_system.index_documents(documents)
    
    # Get and display stats
    stats = rag_system.get_collection_stats()
    logger.info(f"Vector store statistics: {stats}")
    
    # Test retrieval
    logger.info("\nTesting retrieval with sample query...")
    test_query = "What are the rules about liability in contracts?"
    results = rag_system.retrieve(test_query, top_k=3)
    
    logger.info(f"\nRetrieved {len(results)} guidelines for query: '{test_query}'")
    for i, guideline in enumerate(results, 1):
        logger.info(f"\n--- Guideline {i} ---")
        logger.info(f"Source: {guideline.source}")
        logger.info(f"Relevance: {guideline.relevance_score:.4f}")
        logger.info(f"Text preview: {guideline.text[:200]}...")
    
    logger.info("\n✅ RAG system setup complete!")
    logger.info(f"Vector store saved to: backend/data/vector_store/")


if __name__ == "__main__":
    main()
