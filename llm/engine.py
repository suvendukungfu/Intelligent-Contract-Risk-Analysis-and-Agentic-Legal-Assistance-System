"""
llm/engine.py
--------------
The LLM reasoning engine upgraded for Principal-level debugging and robustness.
Supports mandatory Step 6 structured output.
"""

import os
import re
import logging
from typing import Dict, List, Optional

from llm.prompts import build_clause_explanation_prompt

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-large")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

_active_tier: Optional[str] = None

def get_llm_explanation(
    clause: str,
    risk_level: str,
    confidence: float,
    triggers: List[str],
    context_chunks: List[str]
) -> Dict[str, str]:
    """
    Principal Engineer Fix: Generates a highly structured legal analysis.
    """
    global _active_tier

    prompt = build_clause_explanation_prompt(clause, risk_level, confidence, triggers, context_chunks)
    logger.debug(f"[DEBUG_LOG] FINAL LLM PROMPT:\n{prompt}")

    # Try tiers
    for tier_fn, tier_name in [
        (_call_huggingface,    "HuggingFace"),
        (_call_ollama,         "Ollama"),
        (_rule_based_fallback, "RuleBased"),
    ]:
        if _active_tier and _active_tier != tier_name and tier_name != "RuleBased":
            continue
        try:
            raw = tier_fn(prompt, clause, triggers, context_chunks)
            if raw:
                _active_tier = tier_name
                logger.info(f"[LLM] Used tier: {tier_name}")
                logger.debug(f"[DEBUG_LOG] RAW LLM OUTPUT:\n{raw}")
                return _parse_llm_response(raw, clause, triggers, context_chunks)
        except Exception as e:
            logger.warning(f"[LLM] Tier {tier_name} failed: {e}")

    return _rule_based_output(clause, triggers, context_chunks)

def _call_huggingface(prompt: str, *args) -> Optional[str]:
    import requests
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY: headers["Authorization"] = f"Bearer {HF_API_KEY}"

    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.3}}
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        return data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
    return None

def _call_ollama(prompt: str, *args) -> Optional[str]:
    import requests
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
    if response.status_code == 200: return response.json().get("response", "")
    return None

def _rule_based_fallback(*args) -> str: return "RULE_BASED"

def _rule_based_output(clause: str, triggers: List[str], context_chunks: List[str]) -> Dict[str, str]:
    """Fallback using the new structured headers."""
    return {
        "summary": "High-risk contractual provision detected.",
        "explanation": f"The clause contains trigger keywords: {', '.join(triggers)}. This creates standard liability exposure.",
        "legal_implications": "May lead to unanticipated financial penalties or operational friction.",
        "mitigation": "Review for mutual obligations and add liability caps."
    }

def _parse_llm_response(raw_text: str, clause: str, triggers: List[str], context_chunks: List[str]) -> Dict[str, str]:
    """Parses based on Step 6 requirements: Summary, Why Risky, Legal Meaning, Suggested Fix."""
    if raw_text == "RULE_BASED": return _rule_based_output(clause, triggers, context_chunks)

    result = {"summary": "", "explanation": "", "legal_implications": "", "mitigation": ""}

    patterns = {
        "summary":            r"Summary:\s*(.*?)(?=Why Risky:|Legal Meaning:|Suggested Fix:|$)",
        "explanation":        r"Why Risky:\s*(.*?)(?=Legal Meaning:|Suggested Fix:|$)",
        "legal_implications": r"Legal Meaning:\s*(.*?)(?=Suggested Fix:|$)",
        "mitigation":         r"Suggested Fix:\s*(.*?)$",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # Fill gaps
    rule = _rule_based_output(clause, triggers, context_chunks)
    for k in result:
        if not result[k]: result[k] = rule[k]

    return result
