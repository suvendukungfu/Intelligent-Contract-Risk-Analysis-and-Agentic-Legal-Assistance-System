"""
models/scoring.py
------------------
Advanced Risk Scoring Engine for Milestone 3.
Calculates a weighted, confidence-adjusted 'Smart Risk Index'.
"""

from typing import List, Dict, Any

# Map qualitative risk levels to base severity weights
RISK_WEIGHTS = {
    "High Risk": 10.0,
    "Medium Risk": 5.0,  # Reserved for future mid-tier
    "Low Risk": 1.0,
    "Unknown": 0.0
}

def calculate_smart_risk_index(ml_results: List[Dict[str, Any]]) -> float:
    """
    Calculates the 'Smart Risk Index' on a 0-10 scale.

    The score is weighted by:
    1. Base risk level severity.
    2. Model confidence (we trust high-confidence predictions more).
    
    Args:
        ml_results: List of dictionaries containing 'risk_level' and 'confidence'.

    Returns:
        float: Normalized score between 0.0 and 10.0
    """
    if not ml_results:
        return 0.0

    total_weighted_score = 0.0
    max_possible_score = 0.0
    
    for result in ml_results:
        level = result.get("risk_level", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        # Base weight based on classification
        base_weight = RISK_WEIGHTS.get(level, 0.0)
        
        # Adjust weight by confidence score
        # Even if the model flags something as High Risk, if confidence is 55%, 
        # it contributes less to the overall risk index than a 99% confident flag.
        adjusted_score = base_weight * confidence
        
        total_weighted_score += adjusted_score
        
        # Maximum possible score for this clause is if it were High Risk w/ 100% confidence
        max_possible_score += RISK_WEIGHTS["High Risk"]

    if max_possible_score == 0:
        return 0.0

    # Normalize to 0-10 scale
    risk_index = (total_weighted_score / max_possible_score) * 10
    
    # Cap at 10.0 and round
    return round(min(risk_index, 10.0), 1)

def get_severity_assessment(risk_index: float) -> str:
    """
    Returns a human-readable severity assessment based on the Smart Risk Index.
    """
    if risk_index >= 7.5:
        return "CRITICAL — Heavy liability or severe one-sided obligations detected. Immediate legal review required."
    elif risk_index >= 4.5:
        return "HIGH — Significant risk provisions found. Negotiation strongly advised before execution."
    elif risk_index >= 2.0:
        return "MODERATE — Some non-standard clauses identified. Routine legal review sufficient."
    else:
        return "LOW — Contract appears standard and balanced. Proceed with standard operational approval."
