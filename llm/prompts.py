"""
llm/prompts.py
---------------
Enterprise-grade LLM Prompt Engineering for Legal AI Debugging.
Enforces Step 2 requirements: Risk alignment and specific structured output.
"""

def build_clause_explanation_prompt(
    clause: str,
    risk_level: str,
    confidence: float,
    triggers: list,
    context_chunks: list
) -> str:
    """
    Principal Engineer Fix: Enforces strict data utilization and structured format.
    """
    context_text = "\n\n".join(
        [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    ) if context_chunks else "No relevant legal context found."

    triggers_text = ", ".join(triggers) if triggers else "None detected"

    prompt = f"""You are a legal AI assistant.

You MUST use:
1. The clause text provided below.
2. The ML risk prediction (Level: {risk_level}, Confidence: {confidence*100:.1f}%).
3. The retrieved legal context from our knowledge base.

== INSTRUCTIONS ==
* You are the EXPLAINER. Your job is to justify the ML Risk Prediction.
* If ML says HIGH risk -> rigidly explain EXACTLY WHY it is high risk using the provided clause and context.
* If ML says LOW risk -> confirm it is safe and state why it is standard.
* NEVER contradict the ML Risk Prediction.
* Refer to specific trigger words (e.g., {triggers_text}).
* DO NOT hallucinate. Provide factual, grounded legal reasoning using ONLY the provided data.

== INPUT DATA ==

CLAUSE TEXT:
"{clause.strip()}"

ML RISK PREDICTION:
Level: {risk_level}
Confidence: {confidence*100:.1f}%
Trigger Words: {triggers_text}

RETRIEVED LEGAL CONTEXT:
{context_text}

== OUTPUT FORMAT (MANDATORY STRUCTURE) ==

Summary:
[One sentence overview]

Why Risky:
[Detailed justification of the ML risk level using context]

Legal Meaning:
[Plain-English explanation of the legal implications]

Suggested Fix:
[Actionable recommendation to mitigate the risk or maintain compliance]
"""
    return prompt

def build_summary_prompt(high_risk_count: int, total_count: int, file_name: str) -> str:
    """Standard executive summary prompt."""
    risk_pct = (high_risk_count / total_count * 100) if total_count > 0 else 0
    return f"""You are a legal operations analyst. Write a professional 3-sentence executive summary 
for a contract analysis report. 

Document: {file_name}
Risk Intensity: {risk_pct:.1f}% ({high_risk_count}/{total_count} high-risk clauses)

Summarize the document risk status and primary legal concerns.
"""
