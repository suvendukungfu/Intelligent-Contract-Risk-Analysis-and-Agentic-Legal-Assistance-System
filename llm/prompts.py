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

    prompt = f"""You are a senior legal assistant specializing in commercial contract review. 
Your role is to analyze contract clauses for legal risk and provide clear, structured analysis.

IMPORTANT INSTRUCTIONS:
- Base your analysis ONLY on the clause provided and the legal context below.
- Do NOT invent case citations, statutes, or legal principles not mentioned in the context.
- If you are uncertain, explicitly state "Insufficient information to provide a definitive analysis."
- Keep each section concise (2-4 sentences maximum).

== CONTRACT CLAUSE ==
{clause.strip()}

== ML RISK ASSESSMENT ==
Risk Level: {risk_level}
Detected Risk Triggers: {triggers_text}

== RETRIEVED LEGAL CONTEXT ==
{context_text}

== YOUR ANALYSIS (complete each section) ==

RISK EXPLANATION:
[Explain what makes this clause risky and what legal exposure it creates.]

LEGAL IMPLICATIONS:
[Describe the practical and legal consequences if this clause is triggered.]

RECOMMENDED MITIGATION:
[Provide 2-3 specific negotiation points or contract edits to reduce risk.]

LEGAL REFERENCE:
[Name the relevant legal principle, clause type, or standard practice from the context above. If none applies, write "General contract law principles."]
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
