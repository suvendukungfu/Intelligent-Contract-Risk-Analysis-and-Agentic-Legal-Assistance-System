"""
agents/nodes.py
---------------
Overhauled for Principal-level debugging and data flow verification.
Implements Step 1, 3, 5, and 7.
"""

import logging
import json
from typing import Dict, Any
from agents.states import ContractState
from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# Agent 1: Segmentation
# ══════════════════════════════════════════════════════════════════

def parse_contract_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Segmentation"
    if not state["raw_text"]:
        state["errors"].append("CRITICAL: Input text is empty. Skipping analysis.")
        return state

    logger.info(f"[{state['current_step']}] Segmenting text...")
    try:
        clauses = segment_clauses(state["raw_text"])
        state["clauses"] = [c for c in clauses if len(c.strip()) > 5]
    except Exception as e:
        state["errors"].append(f"Segmentation Error: {e}")
        state["clauses"] = [state["raw_text"]]
    return state

# ══════════════════════════════════════════════════════════════════
# Agent 2: ML Risk Detection
# ══════════════════════════════════════════════════════════════════

def risk_detection_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Risk Detection"
    logger.info(f"[{state['current_step']}] Running ML + Anomaly pipeline...")
    
    # 1. Anomaly Detection
    try:
        from models.anomaly_detection import detect_semantic_anomalies
        state["anomalies"] = detect_semantic_anomalies(state["clauses"])
    except Exception as e:
        logger.error(f"[DEBUG] Anomaly detection failed: {e}")
        state["anomalies"] = [{"is_anomaly": False, "anomaly_score": 0.0} for _ in state["clauses"]]

    # 2. Risk Detection
    risks = []
    for idx, clause in enumerate(state["clauses"]):
        level, conf, triggers = risk_engine.analyze_clause(clause)
        
        # Step 7: LOG ML Prediction
        logger.info(f"[DEBUG_LOG] ML CLASSIFY: Index {idx} | Level {level} | Confidence {conf:.2f} | Triggers: {triggers}")

        is_anom = state["anomalies"][idx]["is_anomaly"]
        if is_anom and level == "Low Risk":
            level = "High Risk"
            triggers.append("SEMANTIC_ANOMALY")

        risks.append({
            "clause_idx": idx, "clause": clause, "risk_level": level,
            "confidence": conf, "triggers": triggers, "is_anomaly": is_anom
        })
    
    state["risks"] = risks
    return state

# ══════════════════════════════════════════════════════════════════
# Agent 3: RAG Retrieval (top_k=3-5)
# ══════════════════════════════════════════════════════════════════

def legal_retrieval_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: RAG Retrieval"
    logger.info(f"[{state['current_step']}] Fetching top 3 relevant chunks with RAG quality validation...")
    
    # Step 5 Validation
    if not state.get("risks"):
        logger.warning("[DEBUG] STOP: No risk scores found before retrieval.")
        return state

    from retrieval.rag_engine import retrieve_context_for_clause
    
    retrieved = []
    for risk in state["risks"]:
        if risk["risk_level"] == "High Risk":
            # Step 3: Use top_k=3 and validate relevance
            chunks = retrieve_context_for_clause(risk["clause"], top_k=3)
            
            # Quality Rule: If retrieved chunks are too short/unusable, rely on contextual fallback
            valid_chunks = [c for c in chunks if len(c.strip()) > 40]
            if not valid_chunks:
                logger.warning(f"[DEBUG_LOG] RAG FETCH: Low relevance. Activating Fallback for Index {risk['clause_idx']}.")
                valid_chunks = ["FALLBACK: Standard legal precedent dictates strong limitation of liability. Verify if clause breaches standard thresholds."]

            logger.info(f"[DEBUG_LOG] RAG FETCH: Index {risk['clause_idx']} found {len(valid_chunks)} chunks.")
            retrieved.append({"clause_idx": risk["clause_idx"], "context": valid_chunks})
        else:
            retrieved.append({"clause_idx": risk["clause_idx"], "context": []})
            
    state["retrieved_context"] = retrieved
    return state

# ══════════════════════════════════════════════════════════════════
# Agent 4: LLM Explanation (Step 1 Integration)
# ══════════════════════════════════════════════════════════════════

def reasoning_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: LLM Explanation"
    logger.info(f"[{state['current_step']}] Generating structured data-driven analysis via Parallel Executor...")
    
    from llm.engine import get_llm_explanation, _rule_based_output
    import concurrent.futures
    
    explanations = []
    context_map = {r["clause_idx"]: r["context"] for r in state["retrieved_context"]}
    
    def _process_single_risk(risk):
        idx = risk["clause_idx"]
        context = context_map.get(idx, [])
        
        # Step 5 Validation
        if not risk["clause"] or len(risk["clause"]) < 5:
            return None

        # PERFORMANCE UPGRADE: Completely bypass expensive LLM calls for Safe clauses!
        if risk["risk_level"] != "High Risk":
            return {
                "clause_idx": idx, 
                "summary": "Standard compliance provision.", 
                "explanation": "No significant risk triggers detected by ML Engine.",
                "legal_implications": "Routine contract mechanics.",
                "mitigation": "No immediate legal action required."
            }

        try:
            expl = get_llm_explanation(
                clause=risk["clause"], 
                risk_level=risk["risk_level"], 
                confidence=risk["confidence"],
                triggers=risk["triggers"], 
                context_chunks=context
            )
            
            # Validation Rule: Force Regenerate if explanation maliciously contradicts ML prediction
            expl_text = expl.get("explanation", "").lower()
            if risk["risk_level"] == "High Risk" and ("low risk" in expl_text or "safe" in expl_text):
                logger.warning(f"[DEBUG_LOG] VALIDATION FAILED: LLM contradicted High-Risk prediction for clause {idx}. Regenerating...")
                expl = _rule_based_output(risk["clause"], risk["triggers"], context)
                
            return {"clause_idx": idx, **expl}
            
        except Exception as e:
            logger.error(f"[DEBUG] LLM Explanation failed for clause {idx}: {e}")
            return {
                "clause_idx": idx, 
                "summary": "Error in reasoning agent.", 
                "explanation": "Insufficient data to provide a grounded legal analysis.",
                "legal_implications": "Unknown", 
                "mitigation": "Manual legal review required."
            }

    # Execute heavily parallel queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(_process_single_risk, state["risks"])
        
    for r in results:
        if r is not None:
            explanations.append(r)
            
    # Sort back to original index
    explanations.sort(key=lambda x: x["clause_idx"])
    state["explanations"] = explanations
    return state

# ══════════════════════════════════════════════════════════════════
# Agent 5: Report Generation
# ══════════════════════════════════════════════════════════════════

def report_generator_agent(state: ContractState) -> ContractState:
    state["current_step"] = "Agent: Report Generation"
    from reports.json_report import build_report
    state["final_report"] = build_report(state)
    logger.info(f"[DEBUG_LOG] FINAL REPORT GENERATED for {state['file_name']}.")
    return state
