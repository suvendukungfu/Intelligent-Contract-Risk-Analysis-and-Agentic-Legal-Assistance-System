import logging
import time

logger = logging.getLogger(__name__)


class MockLLM:
    def __init__(self):
        logger.info("Initialized Mock LLM (Gemini API not available)")
    
    def generate_text(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        if "summarize" in prompt_lower and "contract" in prompt_lower:
            return ("This is a Non-Disclosure Agreement between Company A (Disclosing Party) "
                   "and Company B (Receiving Party), establishing confidentiality obligations "
                   "for proprietary information shared between the parties.")
        
        if ("identify" in prompt_lower or "specific legal risk" in prompt_lower) and "clause" in prompt_lower:
            if "indemnification" in prompt_lower or "indemnify" in prompt_lower:
                return ("This clause creates one-sided indemnification where the Receiving Party "
                       "bears all liability, which could expose them to unlimited financial risk.")
            elif "time period" in prompt_lower or "survive" in prompt_lower:
                return ("The confidentiality obligations have no time limit and survive indefinitely, "
                       "which is unusually restrictive and may be unenforceable.")
            elif "confidential" in prompt_lower:
                return ("The definition of confidential information is overly broad and could "
                       "restrict legitimate business activities.")
            else:
                return "This clause contains potentially unfavorable terms that could create legal or financial risk."
        
        if "respond with only one word" in prompt_lower or ("severity" in prompt_lower and ("high, medium, or low" in prompt_lower or "high" in prompt_lower or "medium" in prompt_lower or "low" in prompt_lower)):
            if "indemnification" in prompt_lower or "indemnify" in prompt_lower:
                return "high"
            elif "time period" in prompt_lower or "survive" in prompt_lower or "indefinitely" in prompt_lower:
                return "medium"
            elif "confidential" in prompt_lower:
                return "medium"
            else:
                return "low"
        
        if "mitigation" in prompt_lower or "suggest" in prompt_lower:
            return """1. Negotiate for mutual indemnification where both parties share liability
2. Add a liability cap limiting exposure to a reasonable amount
3. Include carve-outs for third-party claims beyond your control
4. Require written notice and opportunity to defend before indemnification applies"""
        
        if "explanation" in prompt_lower and "consequences" in prompt_lower:
            explanation = ("This clause requires you to compensate the other party for any losses, "
                          "which means you could be financially responsible for damages even if they "
                          "weren't directly your fault.")
            consequences = ("If a claim arises, you could face significant legal costs and damages "
                           "without any cap on your liability, potentially threatening your business's "
                           "financial stability.")
            return f"EXPLANATION: {explanation}\nCONSEQUENCES: {consequences}"
        
        return "This requires careful legal review to assess potential risks and implications."
    
    def __call__(self, prompt: str) -> str:
        time.sleep(0.1)
        return self.generate_text(prompt)
