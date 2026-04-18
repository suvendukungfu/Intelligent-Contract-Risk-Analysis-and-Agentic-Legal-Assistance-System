"""
agents/workflow.py
------------------
Graph orchestration for the LexIQ Agentic AI pipeline.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_graph = None

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
        from agents.states import ContractState
        from agents.nodes import (
            parse_contract_agent,
            risk_detection_agent,
            legal_retrieval_agent,
            reasoning_agent,
            report_generator_agent,
        )

        builder = StateGraph(ContractState)

        builder.add_node("parse",           parse_contract_agent)
        builder.add_node("detect",          risk_detection_agent)
        builder.add_node("retrieve",        legal_retrieval_agent)
        builder.add_node("reason",          reasoning_agent)
        builder.add_node("report",          report_generator_agent)

        builder.set_entry_point("parse")
        builder.add_edge("parse",      "detect")
        builder.add_edge("detect",     "retrieve")
        builder.add_edge("retrieve",   "reason")
        builder.add_edge("reason",     "report")
        builder.add_edge("report",     END)

        return builder.compile()

    except Exception as e:
        logger.error(f"LangGraph Build error: {e}")
        return None

def run_agent_pipeline(raw_text: str, file_name: str = "contract.txt") -> dict:
    global _graph
    if _graph is None:
        _graph = _build_graph()

    initial_state = {
        "raw_text": raw_text,
        "file_name": file_name,
        "clauses": [],
        "risks": [],
        "anomalies": [],
        "retrieved_context": [],
        "explanations": [],
        "final_report": {},
        "errors": [],
        "current_step": "INIT"
    }

    if _graph is not None:
        return _graph.invoke(initial_state)
    else:
        # Fallback sequential
        from agents.nodes import (
            parse_contract_agent,
            risk_detection_agent,
            legal_retrieval_agent,
            reasoning_agent,
            report_generator_agent,
        )
        s = parse_contract_agent(initial_state)
        s = risk_detection_agent(s)
        s = legal_retrieval_agent(s)
        s = reasoning_agent(s)
        s = report_generator_agent(s)
        return s
