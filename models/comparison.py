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
    """
    logger.info(f"[Compare] Comparing '{name_a}' vs '{name_b}'...")

    # 1. Topic & Protection Extraction
    def _extract_topics(clauses):
        topics = {}
        # Domain mapping for standard legal protections
        patterns = {
            "Indemnification": ["indemnif", "hold harmless"],
            "Limitation of Liability": ["liabil", "damages limit", "cap"],
            "Termination": ["terminat", "break clause"],
            "Confidentiality": ["confidential", "non-disclosure", "nda"],
            "Warranties": ["warrant", "guarantee"],
            "Governing Law": ["govern", "jurisdiction", "dispute"],
            "Force Majeure": ["force majeure", "act of god"],
            "Intellectual Property": ["intellectual property", "ip right", "ownership", "copyright"],
            "Data Privacy": ["data protect", "gdpr", "privacy", "personal data"]
        }
        for name, keywords in patterns.items():
            for c in clauses:
                if any(k in c.lower() for k in keywords):
                    topics[name] = c[:100] + "..." # Store snippet
                    break
        return topics

    topics_a = _extract_topics(clauses_a)
    topics_b = _extract_topics(clauses_b)

    missing_in_a = set(topics_b.keys()) - set(topics_a.keys())
    missing_in_b = set(topics_a.keys()) - set(topics_b.keys())

    # 2. Semantic Clause Alignment & Conflict Detection
    alignment_score = 0.0
    mapped_diffs = []
    
    if USE_SEMANTIC and clauses_a and clauses_b:
        try:
            emb_a = _semantic_model.encode(clauses_a, convert_to_tensor=True)
            emb_b = _semantic_model.encode(clauses_b, convert_to_tensor=True)
            cosine_scores = util.cos_sim(emb_a, emb_b)
            
            # Find best matches for high-risk clauses in A
            for idx_a, (clause_a, risk_a) in enumerate(zip(clauses_a, risks_a)):
                if risk_a.get("risk_level") == "High Risk":
                    best_match_idx = int(cosine_scores[idx_a].argmax())
                    sim = float(cosine_scores[idx_a][best_match_idx])
                    
                    if sim > 0.6: # Found a semantic counterpart
                        clause_b = clauses_b[best_match_idx]
                        risk_b = risks_b[best_match_idx]
                        
                        # Detect conflict: A is risky, B is safe/less risky
                        if risk_b.get("risk_level") != "High Risk":
                            mapped_diffs.append({
                                "topic": "Variant Protection",
                                "clause_a": clause_a[:200],
                                "risk_a": "High",
                                "clause_b": clause_b[:200],
                                "risk_b": "Low",
                                "insight": "Contract B uses more balanced terminology in this section."
                            })
            
            best_matches = cosine_scores.max(dim=1)[0]
            alignment_score = float(best_matches.mean().item()) * 100
        except Exception as e:
            logger.error(f"[Compare] Semantic mapping failed: {e}")

    # 3. Final Recommendation Logic
    recommendation = ""
    score_diff = score_a - score_b
    protect_diff = len(missing_in_a) - len(missing_in_b)
    
    # Selection algorithm: Lower score is better, lower missing protections is better
    if abs(score_diff) < 0.5 and protect_diff == 0:
        recommendation = "Both contracts are structurally similar and balanced. Selection can be based on commercial terms rather than legal risk."
    elif score_a < score_b and len(missing_in_a) <= len(missing_in_b):
        recommendation = f"Contract A ('{name_a}') is the preferred choice. It has a lower Risk Index ({score_a} vs {score_b}) and more comprehensive protection coverage."
    elif score_b < score_a and len(missing_in_b) <= len(missing_in_a):
        recommendation = f"Contract B ('{name_b}') is significantly safer. It provides better standard protections and avoids the high-risk formulations found in Contract A."
    else:
        # Complex case: one has lower risk, but more missing protections
        better = name_a if score_a < score_b else name_b
        recommendation = f"Mixed Result: {better} has a lower mathematical risk score, but there are structural gaps in standard protections. Legal manual review is advised."

    return {
        "contract_a": name_a, "contract_b": name_b,
        "score_a": score_a, "score_b": score_b,
        "missing_in_a": list(missing_in_a),
        "missing_in_b": list(missing_in_b),
        "mapped_diffs": mapped_diffs,
        "semantic_alignment": f"{alignment_score:.1f}%" if alignment_score > 0 else "N/A",
        "summary": _build_compare_summary(name_a, name_b, score_a, score_b, missing_in_a, missing_in_b),
        "recommendation": recommendation,
        "high_risk_diff": sum(1 for r in risks_b if r.get("risk_level") == "High Risk") - sum(1 for r in risks_a if r.get("risk_level") == "High Risk")
    }

def _build_compare_summary(name_a, name_b, score_a, score_b, missing_a, missing_b):
    """Basic structural summary."""
    summary = f"Comparison between '{name_a}' (Risk: {score_a}) and '{name_b}' (Risk: {score_b}). "
    if missing_a:
        summary += f"Contract A lacks {len(missing_a)} standard protections present in B. "
    if missing_b:
        summary += f"Contract B is missing {len(missing_b)} clauses found in A. "
    return summary
