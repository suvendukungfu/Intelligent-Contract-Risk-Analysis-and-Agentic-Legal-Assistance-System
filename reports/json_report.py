"""
reports/json_report.py
-----------------------
Enforces the mandatory JSON schema for the LexIQ Agentic AI pipeline.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the mandatory structured legal risk report.
    """
    risks_state   = state.get("risks", [])
    explanations = state.get("explanations", [])
    file_name    = state.get("file_name", "Unknown Document")
    
    # Map explanations for easy lookup
    exp_map = {e["clause_idx"]: e for e in explanations}
    
    # 1. Mandatory RISKS list
    structured_risks = []
    high_labels = []
    
    for r in risks_state:
        if r["risk_level"] == "High Risk":
            idx = r["clause_idx"]
            exp = exp_map.get(idx, {})
            
            structured_risks.append({
                "clause": r["clause"],
                "severity": "Critical" if r.get("is_anomaly") else "High",
                "reason": exp.get("explanation", "Potential liability trap detected via ML triggers."),
                "fix": exp.get("mitigation", "Consult legal counsel for specific phrasing adjustments.")
            })
            
            # Collect topics for recommendations
            ref = exp.get("legal_reference", "")
            if ref and ref != "N/A": high_labels.append(ref.split("—")[0].strip())

    # 2. Summary
    high_count = len(structured_risks)
    summary = (
        f"LexIQ Agentic Analysis of '{file_name}'. "
        f"The pipeline identified {high_count} critical risks requiring immediate attention. "
        f"Overall risk posture: {'UNSAFE' if high_count > 3 else 'MODERATE'}."
    )

    # 3. Recommendations
    recommendations = []
    if high_count > 0:
        recommendations.append("Prioritize renegotiation of 'Critical' severity clauses.")
        if high_labels:
            recommendations.append(f"Focus on {', '.join(list(set(high_labels))[:3])} protections.")
    else:
        recommendations.append("No critical high-risk clauses detected. Proceed with standard review.")

    # MANDATORY SCHEMA ENFORCEMENT
    report = {
        "summary": summary,
        "risks": structured_risks,
        "recommendations": recommendations,
        "disclaimer": "AI-Generated Report. LexIQ analysis is not a substitute for professional legal advice."
    }

    return report

def report_to_json_string(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
