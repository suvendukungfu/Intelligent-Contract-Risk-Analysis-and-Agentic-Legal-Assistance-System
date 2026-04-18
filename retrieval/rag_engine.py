"""
retrieval/rag_engine.py
------------------------
The public RAG interface used by the agent pipeline.
Wraps the vector store query and formats results for downstream use.
"""

import logging
from typing import List

from retrieval.vector_store import query_collection

logger = logging.getLogger(__name__)


def retrieve_context_for_clause(clause_text: str, top_k: int = 2) -> List[str]:
    """
    Retrieves the most legally relevant knowledge chunks for a given clause.

    The function handles graceful degradation:
    - If the vector store returns empty → returns a sentinel string
    - If an exception occurs → returns a sentinel string (never crashes)

    Args:
        clause_text: The clause to retrieve context for.
        top_k:       Number of knowledge chunks to retrieve.

    Returns:
        List of relevant legal knowledge strings (max top_k entries).
    """
    if not clause_text or len(clause_text.strip()) < 10:
        return ["Ambiguous clause — insufficient text for context retrieval."]

    try:
        chunks = query_collection(clause_text, top_k=top_k)
        logger.debug(f"[RAG] Retrieved {len(chunks)} chunks for clause snippet: "
                     f"'{clause_text[:60]}...'")
        return chunks
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return ["No relevant legal context found."]
