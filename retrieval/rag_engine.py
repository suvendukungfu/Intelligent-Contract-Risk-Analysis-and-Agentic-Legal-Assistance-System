"""
retrieval/rag_engine.py
------------------------
Hybrid RAG Engine (Milestone 4).
Fuses Dense Semantic Search (ChromaDB) with Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Dict, Any

from retrieval.vector_store import query_collection
from retrieval.knowledge_base import LEGAL_KNOWLEDGE_BASE

try:
    from rank_bm25 import BM25Okapi
    # Initialize BM25 Sparse Index on load
    _corpus = [entry["content"] for entry in LEGAL_KNOWLEDGE_BASE]
    _tokenized_corpus = [doc.lower().split(" ") for doc in _corpus]
    _bm25 = BM25Okapi(_tokenized_corpus)
    USE_HYBRID = True
except ImportError:
    USE_HYBRID = False

logger = logging.getLogger(__name__)

def retrieve_context_for_clause(clause_text: str, top_k: int = 2) -> List[str]:
    """
    Retrieves knowledge chunks using Reciprocal Rank Fusion (RRF).
    """
    if not clause_text or len(clause_text.strip()) < 10:
        return ["Ambiguous clause — insufficient text for context retrieval."]

    try:
        # 1. DENSE SEARCH (ChromaDB)
        dense_chunks = query_collection(clause_text, top_k=top_k*2)
        
        # If hybrid disabled, just return dense
        if not USE_HYBRID:
            return dense_chunks[:top_k]

        # 2. SPARSE SEARCH (BM25)
        tokenized_query = clause_text.lower().split(" ")
        bm25_scores = _bm25.get_scores(tokenized_query)
        # Get top indices
        top_sparse_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]
        sparse_chunks = [_corpus[i] for i in top_sparse_idx]

        # 3. RECIPROCAL RANK FUSION (RRF)
        # RRF_Score = 1 / (k + rank)
        rrf_k = 60
        fused_scores: Dict[str, float] = {}

        for rank, chunk in enumerate(dense_chunks):
            fused_scores[chunk] = fused_scores.get(chunk, 0.0) + (1.0 / (rrf_k + rank))

        for rank, chunk in enumerate(sparse_chunks):
            fused_scores[chunk] = fused_scores.get(chunk, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by fused score
        ranked_chunks = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        final_chunks = [chunk for chunk, score in ranked_chunks[:top_k]]

        logger.debug(f"[RAG] Hybrid retrieval fused {len(final_chunks)} chunks using RRF.")
        return final_chunks

    except Exception as e:
        logger.error(f"[RAG] Hybrid retrieval failed: {e}")
        return ["No relevant legal context found."]
