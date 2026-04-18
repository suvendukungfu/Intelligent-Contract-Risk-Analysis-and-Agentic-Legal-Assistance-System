"""
agents/nodes.py
---------------
Multi-agent architecture for the LexIQ Agentic AI pipeline.
Each agent specialized in one domain (Risk, RAG, Reasoning, Reporting).
"""

import logging
from typing import Dict, Any

from agents.states import ContractState
from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
    # Agent 1: Parsing Agent
# ══════════════════════════════════════════════════════════════════

def parse_contract_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Parsing"
    logger.info(f"[{state['current_step']}] Segmenting text...")
    try:
        clauses = segment_clauses(state["raw_text"])
        state["clauses"] = clauses if clauses else [state["raw_text"]]
    except Exception as e:
        state["errors"].append(f"Parsing Error: {e}")
        state["clauses"] = [state["raw_text"]]
    return state

# ══════════════════════════════════════════════════════════════════
    # Agent 2: Risk Detection Agent (ML)
# ══════════════════════════════════════════════════════════════════

def risk_detection_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Risk Detection"
    logger.info(f"[{state['current_step']}] Running ML Inference...")
    
    try:
        from models.anomaly_detection import detect_semantic_anomalies
        anom_results = detect_semantic_anomalies(state["clauses"])
        state["anomalies"] = anom_results
    except Exception as e:
        state["errors"].append(f"Anomaly Error: {e}")
        state["anomalies"] = [{"is_anomaly": False, "anomaly_score": 0.0} for _ in state["clauses"]]

    risks = []
    for idx, clause in enumerate(state["clauses"]):
        level, conf, triggers = risk_engine.analyze_clause(clause)
        
        # Cross-reference with anomalies
        is_anom = state["anomalies"][idx]["is_anomaly"]
        if is_anom and level == "Low Risk":
            level = "High Risk"
            triggers.append("SEMANTIC_ANOMALY")

        risks.append({
            "clause_idx": idx,
            "clause": clause,
            "risk_level": level,
            "confidence": conf,
            "triggers": triggers,
            "is_anomaly": is_anom
        })
    
    state["risks"] = risks
    return state

# ══════════════════════════════════════════════════════════════════
    # Agent 3: Legal Retrieval Agent (RAG)
# ══════════════════════════════════════════════════════════════════

def legal_retrieval_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Legal Retrieval"
    logger.info(f"[{state['current_step']}] Fetching legal context...")
    
    from retrieval.rag_engine import retrieve_context_for_clause
    
    retrieved = []
    for risk in state["risks"]:
        if risk["risk_level"] == "High Risk":
            try:
                chunks = retrieve_context_for_clause(risk["clause"], top_k=2)
                retrieved.append({"clause_idx": risk["clause_idx"], "context": chunks})
            except:
                retrieved.append({"clause_idx": risk["clause_idx"], "context": []})
        else:
            retrieved.append({"clause_idx": risk["clause_idx"], "context": []})
            
    state["retrieved_context"] = retrieved
    return state

# ══════════════════════════════════════════════════════════════════
    # Agent 4: Reasoning Agent (LLM)
# ══════════════════════════════════════════════════════════════════

def reasoning_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Reasoning"
    logger.info(f"[{state['current_step']}] Logical analysis...")
    
    from llm.engine import get_llm_explanation
    
    explanations = []
    context_map = {r["clause_idx"]: r["context"] for r in state["retrieved_context"]}
    
    for risk in state["risks"]:
        idx = risk["clause_idx"]
        
        # Logic 1: Decision Logic - Confidence Threshold
        if risk["confidence"] < 0.6 and risk["risk_level"] == "High Risk":
            explanations.append({
                "clause_idx": idx,
                "explanation": "LOW_CONFIDENCE_FALLBACK: The model detected a possible risk but with low certainty. Deep manual review is suggested.",
                "mitigation": "Clarify intent with counterparty.",
                "legal_reference": "N/A"
            })
            continue

        if risk["risk_level"] == "High Risk":
            context = context_map.get(idx, [])
            
            # Logic 2: RAG Hit Logic
            if not context or "No relevant legal context found" in str(context):
                explanations.append({
                    "clause_idx": idx, 
                    "explanation": "RAG_MISS_FALLBACK: No specific legal precedent found in knowledge base. Applying general contract principles.",
                    "mitigation": "Standardize using industry-wide templates.",
                    "legal_reference": "General Commercial Practice"
                })
            else:
                try:
                    expl = get_llm_explanation(risk["clause"], risk["risk_level"], risk["triggers"], context)
                    explanations.append({"clause_idx": idx, **expl})
                except:
                    explanations.append({"clause_idx": idx, "explanation": "LLM_ERROR", "mitigation": "Error", "legal_reference": "N/A"})
        else:
            explanations.append({"clause_idx": idx, "explanation": "Low risk. Standard formulation.", "mitigation": "None", "legal_reference": "N/A"})
            
    state["explanations"] = explanations
    return state

# ══════════════════════════════════════════════════════════════════
    # Agent 5: Report Generator Agent
# ══════════════════════════════════════════════════════════════════

def report_generator_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Reporting"
    logger.info(f"[{state['current_step']}] Finalizing JSON output...")
    
    from reports.json_report import build_report
    state["final_report"] = build_report(state)
    return state
