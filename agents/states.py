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

    # ── AGENT STATE ────────────────────────────────────────────────────────
    clauses: List[str]                   # List of individual legal clauses
    risks: List[Dict[str, Any]]          # Agent 1 (RiskDetection): ML results
    anomalies: List[Dict[str, Any]]      # Agent 1 (RiskDetection): Semantic outliers
    retrieved_context: List[Dict[str, Any]] # Agent 2 (LegalRetrieval): RAG results
    explanations: List[Dict[str, Any]]   # Agent 3 (Reasoning): LLM logic
    final_report: Dict[str, Any]         # Agent 4 (Report): Structured JSON

    # ── META ───────────────────────────────────────────────────────────────
    errors: List[str]                    # Any non-fatal errors collected during pipeline
    current_step: str                    # Tracks which agent is currently active

