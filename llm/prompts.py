"""
llm/prompts.py
---------------
All LLM prompt templates for the Legal AI system.
Centralising prompts here makes them easy to iterate and A/B test.
"""


def build_clause_explanation_prompt(
    clause: str,
    risk_level: str,
    triggers: list,
    context_chunks: list
) -> str:
    """
    Builds the main reasoning prompt sent to the LLM for each high-risk clause.

    Design principles:
    - Role injection: frames the model as a legal assistant
    - Grounded context: only the retrieved legal knowledge is provided (no hallucination)
    - Structured output: clearly labels the sections to extract
    - Uncertainty guard: explicitly instructs the model to admit uncertainty
    """
    context_text = "\n\n".join(
        [f"[Legal Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    ) if context_chunks else "[Legal Context]: No relevant legal context retrieved."

    triggers_text = ", ".join(triggers) if triggers else "none identified"

    prompt = f"""You are a senior legal reasoning agent.
Your goal is to analyze specific contract clauses based EXCLUSIVELY on provided legal context.

RULES:
1. ANTI-HALLUCINATION: If the 'RETRIEVED LEGAL CONTEXT' does not contain enough information to analyze the clause, DO NOT guess. You MUST say: "Insufficient data to provide a grounded legal analysis."
2. NO EXTERNAL DATA: Do not use your own training data about laws. Only use the provided segments.
3. CONCISENESS: Limit each section to 2 sentences.

== CONTRACT CLAUSE ==
{clause.strip()}

== ML CLASSIFICATION ==
Risk Level: {risk_level}
Triggers: {triggers_text}

== RETRIEVED LEGAL CONTEXT ==
{context_text}

== MANDATORY STRUCTURE ==

RISK EXPLANATION:
[Explain the risk ONLY if grounded in context. Otherwise say 'insufficient data']

LEGAL IMPLICATIONS:
[Consequences based on context]

RECOMMENDED MITIGATION:
[Specific edit or strategy]

LEGAL REFERENCE:
[The exact segment from context used]
"""
    return prompt


def build_summary_prompt(high_risk_count: int, total_count: int, file_name: str) -> str:
    """
    Prompt for generating the executive contract summary.
    """
    risk_pct = (high_risk_count / total_count * 100) if total_count > 0 else 0
    return f"""You are a legal operations analyst. Write a 3-sentence executive summary 
for a contract analysis report with the following statistics:

Document: {file_name}
Total clauses analyzed: {total_count}
High-risk clauses: {high_risk_count} ({risk_pct:.1f}%)

The summary should:
1. State the overall risk level (Low / Medium / High)
2. Highlight the primary areas of concern
3. Recommend the next action (e.g., "recommend legal review before signing")

Keep it professional and concise. Do not use bullet points.
"""
