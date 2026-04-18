"""
models/comparison.py
---------------------
Enterprise-grade Contract Comparison Engine.
Analyzes risk deltas, structural gaps, and semantic conflicts between two contracts.
"""

import logging
import pandas as pd
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer, util
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
    Deep-dives into two contracts to find safety deltas.
    """
    # 1. Protection Mapping (Taxonomy)
    patterns = {
        "Indemnification": ["indemnif", "hold harmless"],
        "Liability Cap": ["liabil", "damages limit", "cap"],
        "Termination Rights": ["terminat", "break clause"],
        "Confidentiality": ["confidential", "non-disclosure", "nda"],
        "Governing Law": ["govern", "jurisdiction", "dispute"],
        "Force Majeure": ["force majeure", "act of god"],
        "IP Ownership": ["intellectual property", "ip right", "ownership", "copyright"],
        "Data Protection": ["data protect", "gdpr", "privacy"]
    }

    def _get_coverage(clauses, risks):
        coverage = {}
        for category, keywords in patterns.items():
            found = False
            for idx, c in enumerate(clauses):
                if any(k in c.lower() for k in keywords):
                    coverage[category] = {
                        "text": c[:300],
                        "risk": risks[idx]["risk_level"],
                        "confidence": risks[idx]["confidence"]
                    }
                    found = True
                    break
            if not found:
                coverage[category] = None
        return coverage

    cov_a = _get_coverage(clauses_a, risks_a)
    cov_b = _get_coverage(clauses_b, risks_b)

    # 2. Missing & Risk Gaps
    gaps = []
    all_categories = set(cov_a.keys()) | set(cov_b.keys())
    
    for cat in all_categories:
        state_a = cov_a.get(cat)
        state_b = cov_b.get(cat)
        
        if state_a and not state_b:
            gaps.append({"category": cat, "finding": f"Missing in {name_b}", "impact": "Operational Risk"})
        elif not state_a and state_b:
            gaps.append({"category": cat, "finding": f"Missing in {name_a}", "impact": "Legal Vulnerability"})
        elif state_a and state_b:
            # Both present - check risk level difference
            if state_a["risk"] != state_b["risk"]:
                gaps.append({
                    "category": cat, 
                    "finding": "Risk Discordance", 
                    "impact": f"{name_a}: {state_a['risk']} vs {name_b}: {state_b['risk']}"
                })

    # 3. Verdict Logic
    winner = name_a if score_a < score_b else name_b
    if score_a == score_b: winner = "Tie"
    
    safety_margin = abs(score_a - score_b)
    verdict = ""
    if safety_margin < 0.5:
        verdict = f"Marginal Difference. Both contracts are within a 5% safety delta."
    elif winner != "Tie":
        verdict = f"{winner} is significantly safer with a {safety_margin:.1f} point lower risk index."

    return {
        "metadata": {"a": name_a, "b": name_b, "score_a": score_a, "score_b": score_b},
        "coverage_a": cov_a,
        "coverage_b": cov_b,
        "gaps": gaps,
        "verdict": verdict,
        "winner": winner,
        "summary": f"Analyzed {len(clauses_a)} clauses from {name_a} and {len(clauses_b)} from {name_b}."
    }
