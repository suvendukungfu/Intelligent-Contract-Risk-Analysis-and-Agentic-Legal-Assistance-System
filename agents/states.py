"""
agents/states.py
----------------
Defines the shared STATE object that flows through the entire LangGraph pipeline.
Think of it as a "contract briefcase" that gets richer data added at each step.
"""

from typing import TypedDict, List, Dict, Any, Optional


class ContractState(TypedDict):
    """
    The single source of truth for one contract analysis session.
    Each agent node reads from and writes to this dictionary.
    """
    # ── INPUT ──────────────────────────────────────────────────────────────
    raw_text: str                        # Full contract text extracted from upload
    file_name: str                       # Original filename (for report header)

    # ── STATE 1 OUTPUT: Parsing ────────────────────────────────────────────
    clauses: List[str]                   # List of individual legal clauses

    # ── STATE 2 OUTPUT: ML Risk Detection ─────────────────────────────────
    ml_results: List[Dict[str, Any]]     # [{clause, risk_level, confidence, triggers}, ...]

    # ── STATE 3 OUTPUT: RAG Retrieval ─────────────────────────────────────
    retrieved_contexts: List[Dict[str, Any]]  # [{clause_idx, context_chunks}, ...]

    # ── STATE 4 OUTPUT: LLM Reasoning ─────────────────────────────────────
    explanations: List[Dict[str, Any]]   # [{clause_idx, explanation, mitigation, legal_ref}, ...]

    # ── STATE 5 OUTPUT: Final Report ──────────────────────────────────────
    final_report: Dict[str, Any]         # The complete structured JSON report

    # ── META ───────────────────────────────────────────────────────────────
    errors: List[str]                    # Any non-fatal errors collected during pipeline
    current_step: str                    # Tracks which state is currently active
