"""
reports/json_report.py
-----------------------
Assembles the structured JSON report from the completed agent state.
Upgraded in Milestone 3 to include Smart Risk Scoring and Executive Summaries.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from models.scoring import calculate_smart_risk_index, get_severity_assessment

logger = logging.getLogger(__name__)

def build_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the complete structured legal risk report.
    """
    ml_results   = state.get("ml_results", [])
    explanations = state.get("explanations", [])
    file_name    = state.get("file_name", "Unknown Document")
    errors       = state.get("errors", [])

    # ── Compute aggregate statistics ─────────────────────────────────────
    total_clauses    = len(ml_results)
    high_risk_count  = sum(1 for r in ml_results if r["risk_level"] == "High Risk")
    low_risk_count   = total_clauses - high_risk_count
    
    # Milestone 3: Smart Risk Scoring
    risk_index = calculate_smart_risk_index(ml_results)
    severity = get_severity_assessment(risk_index)

    avg_confidence   = (
        sum(r["confidence"] for r in ml_results) / total_clauses
        if total_clauses > 0 else 0
    )

    # ── High-Level Business Summary (Executive Summary) ───────────────────
    contract_summary = (
        f"EXECUTIVE SUMMARY:\n"
        f"This report constitutes an automated intelligence analysis of the document '{file_name}'. "
        f"The AI engine parsed {total_clauses} independent clauses, flagging {high_risk_count} "
        f"({high_risk_count/total_clauses*100:.1f}%) as carrying elevated legal or financial risk. "
        f"The overall Smart Risk Index—weighted for confidence and severity—is {risk_index}/10. "
        f"Classification: {severity}"
    ) if total_clauses > 0 else "No clauses could be extracted from the document."

    # ── Build identified_risks list ───────────────────────────────────────
    exp_map = {e["clause_idx"]: e for e in explanations}
    identified_risks = []

    for result in ml_results:
        idx = result["clause_idx"]
        explanation_data = exp_map.get(idx, {})

        identified_risks.append({
            "clause_number": idx + 1,
            "clause": result["clause"][:500],   
            "risk_level": result["risk_level"],
            "confidence": f"{result['confidence']*100:.1f}%",
            "linguistic_triggers": result.get("triggers", []),
            "is_anomaly": result.get("is_anomaly", False),
            "anomaly_score": result.get("anomaly_score", 0.0),
            "xai_weights": result.get("xai_weights", {}),
            "explanation": explanation_data.get(
                "explanation",
                "Standard clause. No significant risk identified."
            ),
            "legal_implications": explanation_data.get(
                "legal_implications",
                "N/A"
            ),
            "mitigation": explanation_data.get(
                "mitigation",
                "No action required."
            ),
            "legal_reference": explanation_data.get(
                "legal_reference",
                "N/A"
            )
        })

    # ── Recommendations ───────────────────────────────────────────────────
    high_risk_topics = _extract_risk_topics(ml_results, explanations)
    recommendations = _build_recommendations(high_risk_count, total_clauses, high_risk_topics, risk_index)

    # ── Final Report ──────────────────────────────────────────────────────
    report = {
        "report_metadata": {
            "document_name":      file_name,
            "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_version":     "LexIQ Milestone 3.0 — Top 1% SaaS Edition",
            "pipeline_errors":    errors if errors else []
        },
        "contract_summary":   contract_summary,
        "statistics": {
            "total_clauses":     total_clauses,
            "high_risk_clauses": high_risk_count,
            "low_risk_clauses":  low_risk_count,
            "risk_index":        risk_index,
            "avg_confidence":    f"{avg_confidence*100:.1f}%",
            "severity":          severity
        },
        "identified_risks":   identified_risks,
        "severity_assessment": severity,
        "recommendations":    recommendations,
        "disclaimer": (
            "⚠️ AI-Generated Report — Not Legal Advice. "
            "This analysis is produced by LexIQ SaaS for informational purposes only. "
            "It does not constitute legal advice."
        )
    }

    logger.info(f"[Report] Built report: {total_clauses} clauses, Risk Index {risk_index}/10.")
    return report

def _extract_risk_topics(ml_results: list, explanations: list) -> list:
    exp_map = {e["clause_idx"]: e for e in explanations}
    topics = []
    for r in ml_results:
        if r["risk_level"] == "High Risk":
            exp = exp_map.get(r["clause_idx"], {})
            ref = exp.get("legal_reference", "")
            if ref and ref != "N/A":
                base_topic = ref.split("—")[0].strip()
                if base_topic not in topics:
                    topics.append(base_topic)
    return topics[:5]

def _build_recommendations(high_count: int, total: int, topics: list, risk_index: float) -> str:
    if total == 0:
        return "Unable to generate recommendations — no clauses were extracted."

    base = f"Strategic Review: {high_count} high-risk provisions detected. "
    
    if risk_index >= 7.5:
        action = "DO NOT EXECUTE. Immediate escalation to legal counsel is mandated."
    elif risk_index >= 4.5:
        action = "Renegotiation phase required. Target identified high-risk clauses prior to signature."
    else:
        action = "Standard executive review cycle. Proceed with standard operational guidelines."

    topic_str = ""
    if topics:
        topic_str = f"\nPriority Negotiation Areas: {'; '.join(topics)}."

    return base + action + topic_str

def report_to_json_string(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
