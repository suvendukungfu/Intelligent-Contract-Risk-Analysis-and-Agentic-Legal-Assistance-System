"""
models/comparison.py
---------------------
Enterprise-grade Contract Comparison Engine.
Analyzes risk deltas, structural gaps, and semantic conflicts between two contracts.
"""

import logging
import pandas as pd
import os
import sys
from types import ModuleType
from typing import List, Dict, Any

# Step 0: Initialize Logger immediately after imports to prevent NameError
logger = logging.getLogger(__name__)

# Principal Engineer Fix: System-wide Transformers/Keras 3 Compatibility
# 1. Force transformers to ignore TensorFlow (we only need the tokenizer/MiniLM)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_ADDTIONAL_DEPENDENCIES"] = "1"

# 2. Mock Keras version to bypass internal major-version checks in transformers
# This prevents 'ValueError: Your currently installed version of Keras is Keras 3...'
try:
    if "keras" not in sys.modules:
        mock_keras = ModuleType("keras")
        # Fake a compatible version string that satisfies 'parse(v).major > 2' check
        mock_keras.__version__ = "2.15.0" 
        sys.modules["keras"] = mock_keras
        # Use log here only if safe, but we've defined it above
        logger.info("[DEBUG] Mocked Keras version to bypass Transformers blocker.")
except Exception as e:
    pass

# 3. Controlled import of Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer, util
    _semantic_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/hf_cache")
    USE_SEMANTIC = True
except Exception as e:
    # Handle the specific Keras 3 ValueError or any other import error
    logger.warning(f"Semantic comparison disabled due to environment issue: {type(e).__name__}: {e}")
    USE_SEMANTIC = False

def compare_contracts(
    name_a: str, clauses_a: List[str], risks_a: List[Dict[str, Any]],
    name_b: str, clauses_b: List[str], risks_b: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Standardized Enterprise Dual-Contract Analysis.
    """
    # 1. Normalize Risk Scores: (High Risk Clauses / Total Clauses) * 10
    def _compute_metrics(clauses, risks):
        total = len(clauses)
        high = sum(1 for r in risks if r["risk_level"] == "High Risk")
        score = round((high / total) * 10, 1) if total > 0 else 0
        
        # Sort and get top risks by confidence (descending)
        top_risks = sorted([r for r in risks if r["risk_level"] == "High Risk"], 
                           key=lambda x: x.get("confidence", 0), reverse=True)[:5]
        
        return {
            "total_clauses": total,
            "high_risk": high,
            "risk_score": score,
            "top_risks": [{"clause": r["clause"][:100] + "...", "confidence": r.get("confidence", 0)} for r in top_risks]
        }
        
    metrics_a = _compute_metrics(clauses_a, risks_a)
    metrics_b = _compute_metrics(clauses_b, risks_b)
    
    score_a = metrics_a["risk_score"]
    score_b = metrics_b["risk_score"]

    # 2. Protection Mapping (Taxonomy)
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
            best_match = None
            for idx, c in enumerate(clauses):
                if any(k in c.lower() for k in keywords):
                    cur_risk = risks[idx]["risk_level"]
                    cur_conf = risks[idx].get("confidence", 0)
                    
                    if not best_match:
                        best_match = {
                            "text": c[:300] + "...",
                            "risk": cur_risk,
                            "confidence": cur_conf
                        }
                    # Escalate if new match is strictly worse (High Risk)
                    elif cur_risk == "High Risk" and best_match["risk"] != "High Risk":
                        best_match = {"text": c[:300] + "...", "risk": cur_risk, "confidence": cur_conf}
                    # Escalate if both are High Risk, but new has tighter Model Confidence
                    elif cur_risk == "High Risk" and best_match["risk"] == "High Risk" and cur_conf > best_match["confidence"]:
                        best_match = {"text": c[:300] + "...", "risk": cur_risk, "confidence": cur_conf}
            
            coverage[category] = best_match
        return coverage

    cov_a = _get_coverage(clauses_a, risks_a)
    cov_b = _get_coverage(clauses_b, risks_b)

    # 3. Missing & Risk Gaps
    gaps = []
    comparison_logic = []
    
    all_categories = set(cov_a.keys()) | set(cov_b.keys())
    for cat in all_categories:
        state_a = cov_a.get(cat)
        state_b = cov_b.get(cat)
        
        if state_a and not state_b:
            gaps.append({"category": cat, "finding": f"Missing in {name_b}", "impact": "Operational Risk"})
            comparison_logic.append(f"{name_a} includes {cat}, while {name_b} is dangerously missing it.")
        elif not state_a and state_b:
            gaps.append({"category": cat, "finding": f"Missing in {name_a}", "impact": "Legal Vulnerability"})
            comparison_logic.append(f"{name_b} protects {cat}, whereas {name_a} exposes you to risk.")
        elif state_a and state_b:
            if state_a["risk"] != state_b["risk"]:
                gaps.append({
                    "category": cat, 
                    "finding": "Risk Discordance", 
                    "impact": f"{name_a}: {state_a['risk']} vs {name_b}: {state_b['risk']}"
                })
                safer = name_a if state_a['risk'] == "Low Risk" else name_b
                comparison_logic.append(f"For {cat}, {safer} provides safer terms.")

    # 4. Final Decision Engine
    winner = name_a if score_a < score_b else name_b
    if score_a == score_b: 
        winner = "Tie"
    
    safety_margin = abs(score_a - score_b)
    if safety_margin < 0.5:
        verdict = f"Marginal Difference. Both contracts present similar risk profiles (Δ {safety_margin:.1f})."
    elif winner != "Tie":
        verdict = f"{winner} is significantly safer due to a {safety_margin:.1f} point lower risk score and better clause balance."
    else:
        verdict = "Both contracts share identical risk scores."

    return {
        "metadata": {
            "name_a": name_a, "name_b": name_b, 
            "score_a": score_a, "score_b": score_b
        },
        "stats_a": metrics_a,
        "stats_b": metrics_b,
        "coverage_a": cov_a,
        "coverage_b": cov_b,
        "gaps": gaps,
        "comparison_summary": comparison_logic,
        "verdict": verdict,
        "winner": winner
    }
