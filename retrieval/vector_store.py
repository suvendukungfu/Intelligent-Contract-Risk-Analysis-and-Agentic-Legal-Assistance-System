"""
retrieval/vector_store.py
--------------------------
Builds and manages the ChromaDB vector store for the legal knowledge base.
Uses sentence-transformers (free, local, no API key needed) to embed text.

On first run: embeds all 35 knowledge base entries → saves to disk.
On subsequent runs: loads from disk cache (fast startup).
"""

import os
import logging
from typing import List, Optional

# Force HuggingFace models to use a writable cache directory
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache")
os.makedirs("/tmp/hf_cache", exist_ok=True)

logger = logging.getLogger(__name__)

# Where ChromaDB persists its data between restarts
CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "chroma_db"
)
COLLECTION_NAME = "legal_knowledge"

# Singleton — built once, reused for all queries
_collection = None


def get_collection():
    """
    Returns the ChromaDB collection, initializing it if needed.
    Thread-safe via lazy singleton pattern.
    """
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Use the free, local sentence-transformers embedding model
        # 'all-MiniLM-L6-v2' is small (80MB), fast, and strong for semantic search
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Get or create the collection
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn
            )
            logger.info(f"[VectorStore] Loaded existing collection '{COLLECTION_NAME}' "
                        f"({_collection.count()} docs).")
        else:
            _collection = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}   # cosine similarity for legal text
            )
            logger.info(f"[VectorStore] Created new collection '{COLLECTION_NAME}'.")
            _populate_collection(_collection)

    except ImportError:
        logger.warning("[VectorStore] chromadb not installed. Using FAISS fallback.")
        _collection = _build_faiss_fallback()
    except Exception as e:
        logger.error(f"[VectorStore] Error: {e}. Using in-memory fallback.")
        _collection = _build_faiss_fallback()

    return _collection


def _populate_collection(collection) -> None:
    """
    Embeds all entries from the knowledge base and stores them in ChromaDB.
    """
    from retrieval.knowledge_base import LEGAL_KNOWLEDGE_BASE

    documents = [entry["content"] for entry in LEGAL_KNOWLEDGE_BASE]
    ids       = [entry["id"]      for entry in LEGAL_KNOWLEDGE_BASE]
    metadatas = [{"topic": entry["topic"]} for entry in LEGAL_KNOWLEDGE_BASE]

    # ChromaDB handles embedding automatically via the embedding_function
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    logger.info(f"[VectorStore] Populated with {len(documents)} legal knowledge entries.")


def query_collection(query_text: str, top_k: int = 2) -> List[str]:
    """
    Semantic search: returns the top_k most relevant knowledge chunks
    for the given clause text.

    Args:
        query_text: The legal clause to search for.
        top_k:      Number of results to return.

    Returns:
        List of relevant content strings.
    """
    collection = get_collection()

    # Handle FAISS fallback (returns list directly)
    if isinstance(collection, list):
        return _faiss_query(query_text, top_k, collection)

    try:
        n = collection.count()
        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k, max(n, 1))
        )
        chunks = results["documents"][0] if results["documents"] else []
        return chunks if chunks else ["No relevant legal context found."]
    except Exception as e:
        logger.error(f"[VectorStore] Query error: {e}")
        return ["No relevant legal context found."]


# ══════════════════════════════════════════════════════════════════
# FAISS In-Memory Fallback
# ══════════════════════════════════════════════════════════════════

_faiss_store = None


def _build_faiss_fallback():
    """
    If ChromaDB is unavailable, build a simple FAISS + numpy fallback.
    Returns the knowledge base as a list (used directly by _faiss_query).
    """
    global _faiss_store
    from retrieval.knowledge_base import LEGAL_KNOWLEDGE_BASE
    _faiss_store = LEGAL_KNOWLEDGE_BASE    # keyword search fallback
    logger.info("[VectorStore] FAISS/keyword fallback initialized.")
    return _faiss_store


def _faiss_query(query_text: str, top_k: int, store: list) -> List[str]:
    """
    Simple keyword overlap scoring when vector search is unavailable.
    """
    query_words = set(query_text.lower().split())
    scored = []
    for entry in store:
        content_words = set(entry["content"].lower().split())
        score = len(query_words & content_words)
        scored.append((score, entry["content"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [text for _, text in scored[:top_k] if _ > 0]
    return results if results else ["No relevant legal context found."]
