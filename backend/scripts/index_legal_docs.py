import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag_system import RAGSystem
from core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_legal_documents(docs_dir: Path) -> list:
    documents = []
    
    for file_path in docs_dir.glob("*.txt"):
        logger.info(f"Loading {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        documents.append({
            "text": text,
            "source": file_path.stem.replace('_', ' ').title(),
            "url": ""
        })
    
    return documents


def main():
    settings = get_settings()
    
    logger.info("Initializing RAG system")
    rag = RAGSystem(vector_store_path=settings.vector_store_path)
    
    stats = rag.get_collection_stats()
    logger.info(f"Current collection stats: {stats}")
    
    if stats['total_chunks'] > 0:
        logger.warning(f"Collection already contains {stats['total_chunks']} chunks")
        response = input("Clear existing collection and re-index? (y/n): ")
        if response.lower() == 'y':
            logger.info("Clearing existing collection")
            rag.clear_collection()
        else:
            logger.info("Keeping existing collection, adding new documents")
    
    docs_dir = Path(__file__).parent.parent / "data" / "legal_documents"
    logger.info(f"Loading documents from {docs_dir}")
    
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return
    
    documents = load_legal_documents(docs_dir)
    logger.info(f"Loaded {len(documents)} documents")
    
    if not documents:
        logger.warning("No documents found to index")
        return
    
    logger.info("Indexing documents into vector store")
    rag.index_documents(documents)
    
    stats = rag.get_collection_stats()
    logger.info(f"Indexing complete! Collection stats: {stats}")
    
    logger.info("\nTesting retrieval with sample query...")
    test_query = "What are the requirements for liability clauses?"
    guidelines = rag.retrieve(test_query, top_k=2)
    
    logger.info(f"\nRetrieved {len(guidelines)} guidelines for query: '{test_query}'")
    for i, guideline in enumerate(guidelines, 1):
        logger.info(f"\n{i}. Source: {guideline.source}")
        logger.info(f"   Relevance: {guideline.relevance_score:.3f}")
        logger.info(f"   Text: {guideline.text[:200]}...")


if __name__ == "__main__":
    main()
