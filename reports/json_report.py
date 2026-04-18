"""
reports/json_report.py
-----------------------
Generates a structured, professional Legal AI Risk Report.
Aligned with Agentic Pipeline Milestone 4.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a professional 6-section legal risk report.
    """
    risks_state   = state.get("risks", [])
    explanations  = state.get("explanations", [])
    file_name     = state.get("file_name", "Unknown Document")
    
    exp_map = {e["clause_idx"]: e for e in explanations}
    
    # 1. Executive Summary Logic
    total_clauses = len(risks_state)
    high_risks = [r for r in risks_state if r["risk_level"] == "High Risk"]
    high_count = len(high_risks)
    
    risk_score = min(10.0, round((high_count / total_clauses * 15) if total_clauses > 0 else 0, 1))
    status = "HIGH RISK" if risk_score >= 7.0 else ("MEDIUM RISK" if risk_score >= 4.0 else "LOW RISK")
    
    executive_summary = {
        "document_name": file_name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_risk_score": f"{risk_score}/10",
        "contract_status": status,
        "summary_statement": f"Automated analysis identified {high_count} high-risk provisions across {total_clauses} clauses. "
                             f"The document shows { 'elevated' if high_count > 2 else 'standard' } liability exposure."
    }

    # 2. Risk Breakdown & 5. Explainability (Integrated)
    risk_breakdown = []
    explainability = []
    
    for r in risks_state:
        idx = r["clause_idx"]
        exp = exp_map.get(idx, {})
        
        entry = {
            "clause_number": idx + 1,
            "severity": "CRITICAL" if r.get("is_anomaly") else r["risk_level"].upper(),
            "clause_text": r["clause"][:300] + "...",
            "detected_triggers": r.get("triggers", [])
        }
        risk_breakdown.append(entry)
        
        if r["risk_level"] == "High Risk":
            explainability.append({
                "clause_number": idx + 1,
                "ml_confidence": f"{r['confidence']*100:.1f}%",
                "reasoning": exp.get("explanation", "Potential hidden liability detected via semantic triggers."),
                "mitigation_strategy": exp.get("mitigation", "Seek express clarification on the scope of internal obligations.")
            })

    # 3. Key Risk Insights (Top 5 Dangerous)
    # Sort high risks by confidence * anomaly_weight
    scored_highs = []
    for r in high_risks:
        score = r["confidence"] * (2.0 if r.get("is_anomaly") else 1.0)
        scored_highs.append((score, r))
    
    scored_highs.sort(key=lambda x: x[0], reverse=True)
    top_5 = [s[1] for s in scored_highs[:5]]
    
    key_insights = []
    for r in top_5:
        idx = r["clause_idx"]
        exp = exp_map.get(idx, {})
        key_insights.append({
            "topic": exp.get("legal_reference", "General Liability") if exp.get("legal_reference") != "N/A" else "Contractual Obligation",
            "clause_snippet": r["clause"][:150] + "...",
            "primary_concern": exp.get("explanation", "Atypical legal formulation detected.")
        })

    # 4. Recommendations
    recommendations = []
    unique_topics = list(set([k["topic"] for k in key_insights]))
    for topic in unique_topics:
        recommendations.append({
            "category": topic,
            "action": f"Renegotiate {topic} boundaries to include a fixed liability cap and mutual indemnification."
        })
    if not recommendations:
        recommendations.append({"category": "General", "action": "Proceed with standard legal review cycle."})

    # 6. Final Report Assembly
    report = {
        "executive_summary": executive_summary,
        "risk_breakdown": risk_breakdown,
        "key_risk_insights": key_insights,
        "recommendations": recommendations,
        "explainability": explainability,
        "disclaimer": "⚠️ AI-Generated Report — Not Legal Advice. LexIQ analysis is for informational purposes only."
    }

    return report

def report_to_json_string(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
