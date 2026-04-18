"""
agents/nodes.py
---------------
The 5 node functions executed in the LangGraph workflow.
Each node takes the ContractState, does work, and returns an updated state.
"""

import logging
import traceback
from typing import Dict, Any

from agents.states import ContractState
from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# STATE 1 — Parse Contract
# ══════════════════════════════════════════════════════════════════

def node_parse_contract(state: ContractState) -> ContractState:
    """
    Splits raw contract text into discrete legal clauses.
    Uses the existing RegEx-based clause segmenter.
    """
    state["current_step"] = "STATE_1: Parsing Contract"
    logger.info(f"[Agent] {state['current_step']}")

    try:
        clauses = segment_clauses(state["raw_text"])
        if not clauses:
            # Fallback: treat whole text as one clause
            clauses = [state["raw_text"].strip()]
            state["errors"].append("Clause segmentation found no markers; treating document as single block.")

        state["clauses"] = clauses
        logger.info(f"[Agent] Segmented into {len(clauses)} clauses.")
    except Exception as e:
        state["errors"].append(f"STATE_1 Error: {str(e)}")
        state["clauses"] = [state["raw_text"].strip()] if state["raw_text"] else []

    return state


# ══════════════════════════════════════════════════════════════════
# STATE 2 — ML Risk Detection
# ══════════════════════════════════════════════════════════════════

def node_detect_risks(state: ContractState) -> ContractState:
    """
    Runs each clause through the existing Logistic Regression classifier.
    Produces risk level, confidence score, and linguistic triggers for each clause.
    """
    state["current_step"] = "STATE_2: ML Risk Detection"
    logger.info(f"[Agent] {state['current_step']}")

    ml_results = []
    for idx, clause in enumerate(state["clauses"]):
        try:
            risk_level, confidence, triggers = risk_engine.analyze_clause(clause)
            ml_results.append({
                "clause_idx": idx,
                "clause": clause,
                "risk_level": risk_level,
                "confidence": round(confidence, 4),
                "triggers": triggers
            })
        except Exception as e:
            state["errors"].append(f"STATE_2 Error on clause {idx}: {str(e)}")
            ml_results.append({
                "clause_idx": idx,
                "clause": clause,
                "risk_level": "Unknown",
                "confidence": 0.0,
                "triggers": []
            })

    state["ml_results"] = ml_results
    high = sum(1 for r in ml_results if r["risk_level"] == "High Risk")
    logger.info(f"[Agent] {high}/{len(ml_results)} clauses flagged as High Risk.")
    return state


# ══════════════════════════════════════════════════════════════════
# STATE 3 — Retrieve Legal Context (RAG)
# ══════════════════════════════════════════════════════════════════

def node_retrieve_context(state: ContractState) -> ContractState:
    """
    For each HIGH-RISK clause, retrieves the top-2 most relevant
    legal knowledge chunks from the ChromaDB vector store.
    Low-risk clauses skip RAG to save compute time.
    """
    state["current_step"] = "STATE_3: RAG Retrieval"
    logger.info(f"[Agent] {state['current_step']}")

    # Import here to avoid circular imports at module load time
    from retrieval.rag_engine import retrieve_context_for_clause

    retrieved = []
    for result in state["ml_results"]:
        if result["risk_level"] == "High Risk":
            try:
                chunks = retrieve_context_for_clause(result["clause"], top_k=2)
                retrieved.append({
                    "clause_idx": result["clause_idx"],
                    "context_chunks": chunks
                })
            except Exception as e:
                state["errors"].append(f"STATE_3 RAG Error clause {result['clause_idx']}: {str(e)}")
                retrieved.append({
                    "clause_idx": result["clause_idx"],
                    "context_chunks": ["No relevant legal context found."]
                })
        else:
            # Low-risk: skip RAG, leave empty
            retrieved.append({
                "clause_idx": result["clause_idx"],
                "context_chunks": []
            })

    state["retrieved_contexts"] = retrieved
    logger.info(f"[Agent] RAG complete for {len(retrieved)} clauses.")
    return state


# ══════════════════════════════════════════════════════════════════
# STATE 4 — LLM Reasoning
# ══════════════════════════════════════════════════════════════════

def node_generate_explanations(state: ContractState) -> ContractState:
    """
    Sends each high-risk clause + its retrieved legal context to the LLM.
    The LLM generates a structured legal explanation with mitigation advice.
    """
    state["current_step"] = "STATE_4: LLM Reasoning"
    logger.info(f"[Agent] {state['current_step']}")

    from llm.engine import get_llm_explanation

    explanations = []
    context_map = {r["clause_idx"]: r["context_chunks"] for r in state["retrieved_contexts"]}

    for result in state["ml_results"]:
        idx = result["clause_idx"]
        if result["risk_level"] == "High Risk":
            try:
                context_chunks = context_map.get(idx, [])
                explanation = get_llm_explanation(
                    clause=result["clause"],
                    risk_level=result["risk_level"],
                    triggers=result["triggers"],
                    context_chunks=context_chunks
                )
                explanations.append({"clause_idx": idx, **explanation})
            except Exception as e:
                state["errors"].append(f"STATE_4 LLM Error clause {idx}: {str(e)}")
                explanations.append({
                    "clause_idx": idx,
                    "explanation": "Insufficient information to analyze this clause.",
                    "mitigation": "Consult a licensed attorney.",
                    "legal_reference": "N/A"
                })
        else:
            # Low-risk clauses get a brief standard note
            explanations.append({
                "clause_idx": idx,
                "explanation": "This clause appears standard and low-risk. No immediate concerns identified.",
                "mitigation": "No action required.",
                "legal_reference": "N/A"
            })

    state["explanations"] = explanations
    logger.info(f"[Agent] LLM explanations generated for {len(explanations)} clauses.")
    return state


# ══════════════════════════════════════════════════════════════════
# STATE 5 — Generate Final Report
# ══════════════════════════════════════════════════════════════════

def node_generate_report(state: ContractState) -> ContractState:
    """
    Assembles all agent outputs into the final structured JSON report.
    This is the "finished product" — ready for display and PDF export.
    """
    state["current_step"] = "STATE_5: Final Report"
    logger.info(f"[Agent] {state['current_step']}")

    from reports.json_report import build_report

    try:
        report = build_report(state)
        state["final_report"] = report
    except Exception as e:
        state["errors"].append(f"STATE_5 Report Error: {str(e)}")
        state["final_report"] = {
            "contract_summary": "Report generation failed.",
            "identified_risks": [],
            "severity_assessment": "Unknown",
            "recommendations": "Manual review required.",
            "disclaimer": "AI-generated. Not legal advice."
        }

    logger.info("[Agent] Pipeline complete.")
    return state
