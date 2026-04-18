"""
reports/json_report.py
-----------------------
Generates a professional 6-section legal risk report.
Aligned with Principal-level structured output.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a professional legal risk report based on Principal Debugger Step 6.
    """
    risks_state   = state.get("risks", [])
    explanations  = state.get("explanations", [])
    file_name     = state.get("file_name", "Unknown Document")
    
    exp_map = {e["clause_idx"]: e for e in explanations}
    
    total_clauses = len(risks_state)
    high_risks = [r for r in risks_state if r["risk_level"] == "High Risk"]
    high_count = len(high_risks)
    
    risk_score = min(10.0, round((high_count / total_clauses * 10) if total_clauses > 0 else 0, 1))
    status = "HIGH RISK" if risk_score >= 7.0 else ("MEDIUM RISK" if risk_score >= 4.0 else "LOW RISK")
    
    executive_summary = {
        "document": file_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "overall_risk_score": f"{risk_score}/10",
        "contract_status": status,
        "summary": f"Detected {high_count} critical risks in {file_name}. Recommendation: {status} Review."
    }

    risk_breakdown = []
    explainability = []
    
    for r in risks_state:
        idx = r["clause_idx"]
        exp = exp_map.get(idx, {})
        
        risk_breakdown.append({
            "idx": idx + 1,
            "level": r["risk_level"],
            "clause": r["clause"][:200] + "...",
            "triggers": r.get("triggers", [])
        })
        
        if r["risk_level"] == "High Risk":
            explainability.append({
                "idx": idx + 1,
                "confidence": f"{r['confidence']*100:.1f}%",
                "summary": exp.get("summary", "N/A"),
                "reason": exp.get("explanation", "Risky formulation."),
                "meaning": exp.get("legal_implications", "Legal exposure."),
                "fix": exp.get("mitigation", "Consult counsel.")
            })

    return {
        "executive_summary": executive_summary,
        "risk_breakdown": risk_breakdown,
        "explainability": explainability,
        "disclaimer": "AI-Generated, not legal advice."
    }
