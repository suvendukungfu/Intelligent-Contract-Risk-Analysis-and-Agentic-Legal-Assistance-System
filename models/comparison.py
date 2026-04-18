"""
models/comparison.py
---------------------
Multi-Contract Comparison Engine.
Compares two parsed contracts (list of clauses) to identify missing protections,
semantic similarities, and structural differences.
"""

import logging
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer, util
    # Load lightweight local semantic model used by our RAG
    _semantic_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/hf_cache")
    USE_SEMANTIC = True
except ImportError:
    USE_SEMANTIC = False

logger = logging.getLogger(__name__)

def compare_contracts(
    name_a: str, clauses_a: List[str], risks_a: List[Dict[str, Any]], score_a: float,
    name_b: str, clauses_b: List[str], risks_b: List[Dict[str, Any]], score_b: float
) -> Dict[str, Any]:
    """
    Compares two contracts and generates an analytical delta.
    
    Returns:
        dict containing score deltas, missing protections, and alignment metrics.
    """
    logger.info(f"[Compare] Comparing '{name_a}' vs '{name_b}'...")

    # Basic Risk Score Delta
    score_delta = score_b - score_a

    # Count high risk clauses
    a_high = sum(1 for r in risks_a if r.get("risk_level") == "High Risk")
    b_high = sum(1 for r in risks_b if r.get("risk_level") == "High Risk")

    # Topic extraction (simple heuristic based on keywords)
    def _extract_topics(clauses, risks):
        topics = set()
        for c, r in zip(clauses, risks):
            text = c.lower()
            if "indemnif" in text: topics.add("Indemnification")
            if "liabil" in text: topics.add("Limitation of Liability")
            if "terminat" in text: topics.add("Termination")
            if "confidential" in text: topics.add("Confidentiality")
            if "warrant" in text: topics.add("Warranties")
            if "govern" in text: topics.add("Governing Law")
            if "force majeure" in text: topics.add("Force Majeure")
        return topics

    topics_a = _extract_topics(clauses_a, risks_a)
    topics_b = _extract_topics(clauses_b, risks_b)

    # Missing protections
    missing_in_a = topics_b - topics_a
    missing_in_b = topics_a - topics_b

    # Semantic Alignment (calculating average similarity)
    alignment_score = 0.0
    if USE_SEMANTIC and clauses_a and clauses_b:
        try:
            # Encode all clauses
            emb_a = _semantic_model.encode(clauses_a, convert_to_tensor=True)
            emb_b = _semantic_model.encode(clauses_b, convert_to_tensor=True)

            # Compute semantic matching (for every clause in A, find best in B)
            cosine_scores = util.cos_sim(emb_a, emb_b)
            # Take max similarity for each clause in A
            best_matches = cosine_scores.max(dim=1)[0]
            alignment_score = float(best_matches.mean().item()) * 100
        except Exception as e:
            logger.error(f"[Compare] Semantic similarity failed: {e}")
            alignment_score = 0.0

    return {
        "contract_a": name_a,
        "contract_b": name_b,
        "score_a": score_a,
        "score_b": score_b,
        "score_delta": round(score_delta, 1),
        "high_risk_diff": b_high - a_high,
        "topics_a": list(topics_a),
        "topics_b": list(topics_b),
        "missing_in_a": list(missing_in_a),
        "missing_in_b": list(missing_in_b),
        "semantic_alignment": f"{alignment_score:.1f}%" if alignment_score > 0 else "N/A",
        "summary": _build_compare_summary(name_a, name_b, score_a, score_b, missing_in_a, missing_in_b)
    }

def _build_compare_summary(name_a: str, name_b: str, score_a: float, score_b: float, missing_a: set, missing_b: set) -> str:
    """Generates the executive summary for the comparison."""
    summary = ""
    
    if abs(score_a - score_b) <= 0.5:
        summary += f"Both '{name_a}' and '{name_b}' present a similar risk profile. "
    elif score_a > score_b:
        summary += f"'{name_a}' is significantly riskier (Score: {score_a}) compared to '{name_b}' (Score: {score_b}). "
    else:
        summary += f"'{name_b}' is significantly riskier (Score: {score_b}) compared to '{name_a}' (Score: {score_a}). "

    if missing_a:
        summary += f"Notably, '{name_a}' completely lacks standard protections found in the reference document: {', '.join(missing_a)}. "

    if not missing_a and not missing_b:
        summary += "Both documents share a structurally similar foundation regarding standard legal protections."

    return summary
