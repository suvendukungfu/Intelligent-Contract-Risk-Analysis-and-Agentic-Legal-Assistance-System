"""
llm/engine.py
--------------
The LLM reasoning engine with a 3-tier fallback strategy:

Tier 1: HuggingFace Inference API (free, no GPU needed)
         → Uses 'mistralai/Mistral-7B-Instruct-v0.1' or 'google/flan-t5-large'
Tier 2: Ollama local server (if running on the machine)
         → Uses 'mistral' or 'llama3' models
Tier 3: Rule-based analytical fallback
         → Generates structured response from ML triggers (no LLM)

This ensures the system NEVER crashes even without internet or GPU.
"""

import os
import re
import logging
from typing import Dict, List, Optional

from llm.prompts import build_clause_explanation_prompt

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-large")
HF_API_KEY = os.environ.get("HF_API_KEY", "")   # Optional: set for higher rate limits
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

# Cache which tier worked so we don't retry failing tiers on every call
_active_tier: Optional[str] = None


def get_llm_explanation(
    clause: str,
    risk_level: str,
    triggers: List[str],
    context_chunks: List[str]
) -> Dict[str, str]:
    """
    Generates a structured legal explanation for a high-risk clause.

    Returns a dict with keys:
      - explanation:     Why the clause is risky
      - legal_implications: What could go wrong
      - mitigation:     How to fix it
      - legal_reference: The relevant legal principle
    """
    global _active_tier

    prompt = build_clause_explanation_prompt(clause, risk_level, triggers, context_chunks)

    # Try each tier in order
    for tier_fn, tier_name in [
        (_call_huggingface,    "HuggingFace"),
        (_call_ollama,         "Ollama"),
        (_rule_based_fallback, "RuleBased"),
    ]:
        if _active_tier and _active_tier != tier_name and tier_name != "RuleBased":
            # Skip tiers we know don't work, but always allow RuleBased
            continue
        try:
            raw = tier_fn(prompt, clause, triggers, context_chunks)
            if raw:
                _active_tier = tier_name
                logger.info(f"[LLM] Used tier: {tier_name}")
                return _parse_llm_response(raw, clause, triggers, context_chunks)
        except Exception as e:
            logger.warning(f"[LLM] Tier {tier_name} failed: {e}")

    # Should never reach here (rule-based always succeeds)
    return _rule_based_output(clause, triggers, context_chunks)


# ══════════════════════════════════════════════════════════════════
# TIER 1 — HuggingFace Inference API (Free)
# ══════════════════════════════════════════════════════════════════

def _call_huggingface(prompt: str, *args) -> Optional[str]:
    """
    Calls the HuggingFace free inference API.
    Rate limit: ~30 requests/min without API key.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    # Try flan-t5-large first (fast, good for structured output)
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.3,          # Lower = less hallucination
            "do_sample": True,
            "return_full_text": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        elif isinstance(data, dict):
            return data.get("generated_text", "")
    elif response.status_code == 503:
        # Model loading — retry with fallback model
        logger.warning("[HF] Model loading (503). Trying mistral-7b...")
        return _call_huggingface_model(
            "mistralai/Mistral-7B-Instruct-v0.1",
            prompt
        )
    else:
        logger.warning(f"[HF] API returned {response.status_code}: {response.text[:200]}")
        return None


def _call_huggingface_model(model_id: str, prompt: str) -> Optional[str]:
    """Helper to call a specific HuggingFace model."""
    import requests
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 400, "temperature": 0.3, "return_full_text": False}
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
    return None


# ══════════════════════════════════════════════════════════════════
# TIER 2 — Ollama Local Server
# ══════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, *args) -> Optional[str]:
    """
    Calls a locally running Ollama server.
    Start with: `ollama serve` + `ollama pull mistral`
    """
    import requests
    import json

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 400}
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        return response.json().get("response", "")
    return None


# ══════════════════════════════════════════════════════════════════
# TIER 3 — Rule-Based Analytical Fallback (No LLM)
# ══════════════════════════════════════════════════════════════════

# Maps known trigger phrases to explanations
TRIGGER_EXPLANATIONS = {
    "indemnif": {
        "topic": "Indemnification",
        "explanation": "This clause contains an indemnification obligation that may expose one party to broad financial liability for claims arising from the other party's actions or the contract's performance.",
        "mitigation": "Negotiate a cap on indemnification liability (e.g., limited to 12 months of fees paid). Seek mutual indemnification and carve out obligations arising from gross negligence or willful misconduct.",
        "reference": "General Indemnification Law; UCC § 2-719"
    },
    "liabil": {
        "topic": "Liability",
        "explanation": "This clause addresses liability exposure, potentially limiting or expanding one party's financial responsibility. Broad liability provisions or insufficient caps create significant financial risk.",
        "mitigation": "Ensure liability is capped at a reasonable amount. Exclude gross negligence and intentional misconduct from liability limits. Negotiate mutual caps.",
        "reference": "Limitation of Liability — Commercial Contract Best Practice"
    },
    "terminat": {
        "topic": "Termination",
        "explanation": "This clause governs when and how the contract can be ended. Broad termination rights without adequate notice or compensation provisions create operational and financial risk.",
        "mitigation": "Require 30–90 days written notice. Define 'cause' specifically. Include provisions for payment of completed work upon termination.",
        "reference": "General Contract Termination Principles"
    },
    "confidential": {
        "topic": "Confidentiality",
        "explanation": "This clause restricts the disclosure of sensitive information. Overly broad definitions or perpetual duration can create impractical obligations.",
        "mitigation": "Define confidential information specifically. Limit duration to 3–5 years. Include standard exceptions (publicly known, independently developed, legally required disclosure).",
        "reference": "UCC and Common Law Trade Secret Protection"
    },
    "arbitrat": {
        "topic": "Dispute Resolution",
        "explanation": "Mandatory arbitration waives jury trial rights and can limit remedies. An arbitration clause with unfavorable venue can impose significant cost and geographic disadvantage.",
        "mitigation": "Specify AAA or JAMS rules. Require neutral venue. Preserve injunctive relief rights in courts for IP and confidentiality breaches.",
        "reference": "Federal Arbitration Act (FAA); AAA Commercial Rules"
    },
    "govern": {
        "topic": "Governing Law",
        "explanation": "The choice of governing law determines which state's or country's legal framework applies to disputes. An unfamiliar jurisdiction can create hidden legal costs.",
        "mitigation": "Choose a neutral, well-developed jurisdiction (e.g., Delaware, New York). If international, specify CISG applicability or exclusion explicitly.",
        "reference": "Restatement (Second) of Conflict of Laws § 187"
    }
}


def _rule_based_fallback(prompt: str, clause: str, triggers: List[str], context_chunks: List[str]) -> str:
    """
    Generates a structured response from trigger analysis without an LLM.
    Always succeeds.
    """
    return "RULE_BASED"


def _rule_based_output(clause: str, triggers: List[str], context_chunks: List[str]) -> Dict[str, str]:
    """
    Builds explanation dict from keyword triggers. No LLM required.
    """
    # Find the best matching rule
    matched_rule = None
    for trigger in triggers:
        for key, rule in TRIGGER_EXPLANATIONS.items():
            if key in trigger.lower():
                matched_rule = rule
                break
        if matched_rule:
            break

    if matched_rule:
        return {
            "explanation": matched_rule["explanation"],
            "legal_implications": (
                f"The presence of '{', '.join(triggers[:3])}' language indicates this clause "
                f"carries significant {matched_rule['topic']} risk that could result in "
                f"substantial financial or operational liability."
            ),
            "mitigation": matched_rule["mitigation"],
            "legal_reference": matched_rule["reference"]
        }

    # Generic fallback if no trigger matched
    context_snippet = context_chunks[0][:200] if context_chunks and context_chunks[0] != "No relevant legal context found." else ""
    return {
        "explanation": (
            "This clause contains language associated with elevated contractual risk. "
            "The specific obligations or restrictions imposed require careful review."
        ),
        "legal_implications": (
            "Without modification, this clause may expose a party to unanticipated "
            "financial liability, restricted operational freedom, or dispute resolution "
            "disadvantages."
        ),
        "mitigation": (
            "Seek clarification on the scope of obligations. "
            "Negotiate caps, cure periods, and mutual obligations. "
            "Consult a licensed attorney before signing."
        ),
        "legal_reference": (
            context_snippet if context_snippet
            else "General commercial contract law principles and best practices."
        )
    }


def _parse_llm_response(
    raw_text: str,
    clause: str,
    triggers: List[str],
    context_chunks: List[str]
) -> Dict[str, str]:
    """
    Parses the LLM's structured response into a clean dictionary.
    Falls back to rule-based output if parsing fails.
    """
    if raw_text == "RULE_BASED":
        return _rule_based_output(clause, triggers, context_chunks)

    result = {
        "explanation": "",
        "legal_implications": "",
        "mitigation": "",
        "legal_reference": ""
    }

    # Extract sections using the section headers from the prompt
    patterns = {
        "explanation":        r"RISK EXPLANATION:\s*(.*?)(?=LEGAL IMPLICATIONS:|RECOMMENDED MITIGATION:|LEGAL REFERENCE:|$)",
        "legal_implications": r"LEGAL IMPLICATIONS:\s*(.*?)(?=RECOMMENDED MITIGATION:|LEGAL REFERENCE:|$)",
        "mitigation":         r"RECOMMENDED MITIGATION:\s*(.*?)(?=LEGAL REFERENCE:|$)",
        "legal_reference":    r"LEGAL REFERENCE:\s*(.*?)$",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            result[key] = value if value else ""

    # If parsing failed (empty results), fall back to rule-based
    if not any(result.values()):
        logger.warning("[LLM Parser] Could not parse LLM output. Using rule-based fallback.")
        return _rule_based_output(clause, triggers, context_chunks)

    # Fill any empty sections with rule-based output
    rule = _rule_based_output(clause, triggers, context_chunks)
    for key in result:
        if not result[key]:
            result[key] = rule.get(key, "Insufficient information.")

    return result
