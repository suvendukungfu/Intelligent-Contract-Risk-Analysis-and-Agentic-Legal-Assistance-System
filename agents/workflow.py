"""
agents/workflow.py
------------------
Defines and compiles the LangGraph StateGraph — the "brain" of Milestone 2.
The pipeline executes linearly: Parse → Detect → Retrieve → Explain → Report.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Import the compiled graph (built once at module load)
_graph = None


def _build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.
    Uses a try/except so the system degrades gracefully
    if langgraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END
        from agents.states import ContractState
        from agents.nodes import (
            node_parse_contract,
            node_detect_risks,
            node_retrieve_context,
            node_generate_explanations,
            node_generate_report,
        )

        # ── Build the graph ──────────────────────────────────────────────
        builder = StateGraph(ContractState)

        # Register each node (name → function)
        builder.add_node("parse_contract",         node_parse_contract)
        builder.add_node("detect_risks",           node_detect_risks)
        builder.add_node("retrieve_context",       node_retrieve_context)
        builder.add_node("generate_explanations",  node_generate_explanations)
        builder.add_node("generate_report",        node_generate_report)

        # ── Wire the edges (linear pipeline) ────────────────────────────
        builder.set_entry_point("parse_contract")
        builder.add_edge("parse_contract",        "detect_risks")
        builder.add_edge("detect_risks",          "retrieve_context")
        builder.add_edge("retrieve_context",      "generate_explanations")
        builder.add_edge("generate_explanations", "generate_report")
        builder.add_edge("generate_report",       END)

        graph = builder.compile()
        logger.info("[LangGraph] Workflow compiled successfully.")
        return graph

    except ImportError as e:
        logger.warning(f"[LangGraph] Not available: {e}. Will run fallback pipeline.")
        return None
    except Exception as e:
        logger.error(f"[LangGraph] Build error: {e}")
        return None


def run_agent_pipeline(raw_text: str, file_name: str = "contract.txt") -> dict:
    """
    Public entry point. Runs the full 5-step agentic pipeline.

    Args:
        raw_text:  Full text of the uploaded legal document.
        file_name: Original filename for the report header.

    Returns:
        The final ContractState dictionary with all results populated.
    """
    global _graph

    # Build graph once (lazy initialization)
    if _graph is None:
        _graph = _build_graph()

    # ── Initialise the state ─────────────────────────────────────────────
    initial_state = {
        "raw_text": raw_text,
        "file_name": file_name,
        "clauses": [],
        "ml_results": [],
        "retrieved_contexts": [],
        "explanations": [],
        "final_report": {},
        "errors": [],
        "current_step": "INIT"
    }

    # ── Run via LangGraph OR fallback ────────────────────────────────────
    if _graph is not None:
        logger.info("[LangGraph] Running compiled graph...")
        final_state = _graph.invoke(initial_state)
    else:
        logger.warning("[LangGraph] Using sequential fallback pipeline...")
        final_state = _fallback_pipeline(initial_state)

    return final_state


def _fallback_pipeline(state: dict) -> dict:
    """
    Runs all 5 nodes sequentially without LangGraph.
    Used when langgraph is not installed or fails to compile.
    """
    from agents.nodes import (
        node_parse_contract,
        node_detect_risks,
        node_retrieve_context,
        node_generate_explanations,
        node_generate_report,
    )

    state = node_parse_contract(state)
    state = node_detect_risks(state)
    state = node_retrieve_context(state)
    state = node_generate_explanations(state)
    state = node_generate_report(state)
    return state
