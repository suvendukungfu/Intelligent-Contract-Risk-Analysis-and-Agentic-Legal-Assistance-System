"""
Agentic Assistant for intelligent contract risk analysis.
Uses LLM with multi-step reasoning to generate comprehensive risk reports.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai

# Import models - handle both direct and module imports
try:
    from backend.api.models import ParsedDocument, Clause, RiskPrediction, RiskReport, Risk
    from backend.core.rag_system import RAGSystem
    from backend.core.config import get_settings
except ImportError:
    from api.models import ParsedDocument, Clause, RiskPrediction, RiskReport, Risk
    from core.rag_system import RAGSystem
    from core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class AnalysisError(Exception):
    """Exception raised when analysis cannot be completed."""
    pass


class AgenticAssistant:
    """
    Agentic AI assistant for contract risk analysis.
    
    Orchestrates multi-step reasoning using LLM to:
    1. Summarize contract
    2. Identify risks (using ML predictions as hints)
    3. Retrieve legal guidelines via RAG
    4. Assess severity
    5. Generate mitigations
    6. Create plain-language explanations
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize agentic assistant.
        
        Args:
            rag_system: RAG system for legal guideline retrieval
        """
        self.rag = rag_system
        self.max_retries = settings.max_retries
        self.timeout = settings.analysis_timeout_seconds
        self.use_mock = False
        
        # Configure Gemini API
        if settings.llm_provider.lower() == "gemini":
            if not settings.gemini_api_key:
                logger.warning("GEMINI_API_KEY not configured, using mock LLM")
                self.use_mock = True
            else:
                try:
                    genai.configure(api_key=settings.gemini_api_key)
                    logger.info(f"Initialized Gemini with model: {settings.llm_model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}, using mock LLM")
                    self.use_mock = True
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
        
        if self.use_mock:
            from core.mock_llm import MockLLM
            self.mock_llm = MockLLM()
    
    def analyze(
        self,
        contract: ParsedDocument,
        clauses: List[Clause],
        ml_predictions: List[RiskPrediction]
    ) -> RiskReport:
        """
        Generate comprehensive risk analysis for a contract.
        
        Args:
            contract: Parsed contract document
            clauses: Segmented clauses
            ml_predictions: ML classification results
            
        Returns:
            Structured RiskReport object
            
        Raises:
            AnalysisError: If analysis cannot be completed within timeout
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting agentic analysis for document {contract.id}")
            
            # Step 1: Summarize contract
            logger.info("Step 1: Summarizing contract")
            contract_summary = self._summarize_contract(contract, clauses)
            
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise AnalysisError("Analysis timeout during contract summarization")
            
            # Step 2: Identify risks using ML predictions as hints
            logger.info("Step 2: Identifying risks")
            identified_risks = self._identify_risks(clauses, ml_predictions)
            
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise AnalysisError("Analysis timeout during risk identification")
            
            # Step 3-6: For each risk, retrieve guidelines and generate detailed analysis
            logger.info(f"Step 3-6: Analyzing {len(identified_risks)} risks")
            analyzed_risks = []
            
            for risk_data in identified_risks:
                # Check timeout before processing each risk
                if time.time() - start_time > self.timeout:
                    logger.warning(f"Timeout reached, returning partial results with {len(analyzed_risks)} risks")
                    break
                
                analyzed_risk = self._analyze_risk(risk_data, clauses)
                analyzed_risks.append(analyzed_risk)
            
            # Determine overall severity
            overall_severity = self._determine_overall_severity(analyzed_risks)
            
            # Create risk report
            report = RiskReport(
                contract_summary=contract_summary,
                identified_risks=analyzed_risks,
                overall_severity=overall_severity
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            raise AnalysisError(f"Failed to complete analysis: {str(e)}")

    
    def _summarize_contract(self, contract: ParsedDocument, clauses: List[Clause]) -> str:
        """
        Generate a concise summary of the contract.
        
        Args:
            contract: Parsed contract document
            clauses: List of clauses
            
        Returns:
            Contract summary string
        """
        # Prepare contract text (truncate if too long)
        contract_text = contract.text[:5000] if len(contract.text) > 5000 else contract.text
        
        prompt = f"""You are a legal contract analyst. Summarize the following contract in 2-3 sentences.
Focus on:
- Type of contract (e.g., employment, NDA, service agreement)
- Main parties involved
- Key obligations and terms

Contract text:
{contract_text}

Provide a concise summary:"""
        
        summary = self._call_llm_with_retry(prompt)
        return summary.strip()
    
    def _identify_risks(
        self,
        clauses: List[Clause],
        ml_predictions: List[RiskPrediction]
    ) -> List[Dict[str, Any]]:
        """
        Identify risky clauses using ML predictions as hints.
        
        Args:
            clauses: List of clauses
            ml_predictions: ML classification results
            
        Returns:
            List of risk data dictionaries
        """
        # Create mapping of clause_id to prediction
        prediction_map = {pred.clause_id: pred for pred in ml_predictions}
        
        # Filter clauses with high or medium risk
        risky_clauses = []
        for clause in clauses:
            pred = prediction_map.get(clause.id)
            if pred and pred.risk_label in ["high_risk", "medium_risk"]:
                risky_clauses.append({
                    "clause": clause,
                    "prediction": pred
                })
        
        # If no risky clauses found by ML, return empty list
        if not risky_clauses:
            logger.info("No risky clauses identified by ML classifier")
            return []
        
        # Use LLM to validate and describe risks
        identified_risks = []
        
        for item in risky_clauses:
            clause = item["clause"]
            prediction = item["prediction"]
            
            prompt = f"""You are a legal risk analyst. Analyze the following contract clause and identify the specific legal risk.

Clause text:
"{clause.text}"

ML Risk Assessment: {prediction.risk_label} (confidence: {prediction.confidence:.2f})

Provide a brief risk description (1-2 sentences) explaining what makes this clause risky:"""
            
            risk_description = self._call_llm_with_retry(prompt)
            
            identified_risks.append({
                "clause_id": clause.id,
                "clause_text": clause.text,
                "risk_description": risk_description.strip(),
                "ml_risk_label": prediction.risk_label,
                "ml_confidence": prediction.confidence
            })
        
        logger.info(f"Identified {len(identified_risks)} risky clauses")
        return identified_risks
    
    def _analyze_risk(self, risk_data: Dict[str, Any], clauses: List[Clause]) -> Risk:
        """
        Perform detailed analysis of a single risk.
        
        Steps:
        - Retrieve legal guidelines via RAG
        - Assess severity
        - Generate mitigation actions
        - Create plain-language explanation
        
        Args:
            risk_data: Risk data dictionary
            clauses: All clauses (for context)
            
        Returns:
            Risk object with complete analysis
        """
        clause_id = risk_data["clause_id"]
        clause_text = risk_data["clause_text"]
        risk_description = risk_data["risk_description"]
        
        # Step 3: Retrieve legal guidelines via RAG
        logger.debug(f"Retrieving legal guidelines for clause {clause_id}")
        guidelines = self.rag.retrieve(risk_description, top_k=settings.rag_top_k)
        
        # Format guidelines for LLM context
        guidelines_text = ""
        legal_guideline_citations = []
        
        if guidelines:
            guidelines_text = "\n\nRelevant Legal Guidelines:\n"
            for i, guideline in enumerate(guidelines, 1):
                guidelines_text += f"{i}. {guideline.text}\n   Source: {guideline.source}\n"
                citation = f"{guideline.source}"
                if guideline.url:
                    citation += f" ({guideline.url})"
                legal_guideline_citations.append(citation)
        else:
            guidelines_text = "\n\nNo specific legal guidelines found in database. Use general legal knowledge.\n"
        
        # Step 4: Assess severity
        severity = self._assess_severity(risk_data, guidelines_text)
        
        # Step 5: Generate mitigation actions
        mitigation_actions = self._generate_mitigations(
            clause_text,
            risk_description,
            severity,
            guidelines_text
        )
        
        # Step 6: Create plain-language explanation
        explanation, consequences = self._generate_explanation(
            clause_text,
            risk_description,
            severity,
            guidelines_text
        )
        
        return Risk(
            clause_id=clause_id,
            clause_text=clause_text,
            risk_description=risk_description,
            severity=severity,
            explanation=explanation,
            consequences=consequences,
            mitigation_actions=mitigation_actions,
            legal_guidelines=legal_guideline_citations
        )

    
    def _assess_severity(self, risk_data: Dict[str, Any], guidelines_text: str) -> str:
        """
        Assess the severity level of a risk.
        
        Args:
            risk_data: Risk data dictionary
            guidelines_text: Retrieved legal guidelines
            
        Returns:
            Severity level: "high", "medium", or "low"
        """
        clause_text = risk_data["clause_text"]
        risk_description = risk_data["risk_description"]
        ml_risk_label = risk_data.get("ml_risk_label", "unknown")
        
        prompt = f"""You are a legal risk analyst. Assess the severity of this contract risk.

Clause: "{clause_text}"

Risk: {risk_description}

ML Assessment: {ml_risk_label}
{guidelines_text}

Severity Criteria:
- HIGH: Could result in significant financial loss, legal liability, or loss of critical rights
- MEDIUM: Could result in moderate financial impact or operational constraints
- LOW: Minor inconvenience or easily manageable risk

Respond with ONLY one word: high, medium, or low"""
        
        severity_response = self._call_llm_with_retry(prompt).strip().lower()
        
        # Validate response
        if severity_response not in ["high", "medium", "low"]:
            # Fallback based on ML prediction
            logger.warning(f"Invalid severity response: {severity_response}, using ML prediction")
            if ml_risk_label == "high_risk":
                return "high"
            elif ml_risk_label == "medium_risk":
                return "medium"
            else:
                return "low"
        
        return severity_response
    
    def _generate_mitigations(
        self,
        clause_text: str,
        risk_description: str,
        severity: str,
        guidelines_text: str
    ) -> List[str]:
        """
        Generate mitigation actions for a risk.
        
        Args:
            clause_text: Clause text
            risk_description: Risk description
            severity: Severity level
            guidelines_text: Retrieved legal guidelines
            
        Returns:
            List of mitigation action strings
        """
        prompt = f"""You are a legal advisor. Suggest practical mitigation actions for this contract risk.

Clause: "{clause_text}"

Risk: {risk_description}
Severity: {severity}
{guidelines_text}

Provide 2-4 specific, actionable mitigation steps. Format as a numbered list:
1. [First mitigation action]
2. [Second mitigation action]
..."""
        
        response = self._call_llm_with_retry(prompt)
        
        # Parse numbered list
        mitigation_actions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Match lines starting with number and period
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                action = line.lstrip('0123456789.-•) ').strip()
                if action:
                    mitigation_actions.append(action)
        
        # Ensure we have at least one mitigation
        if not mitigation_actions:
            mitigation_actions = ["Consult with a legal professional to review and revise this clause"]
        
        return mitigation_actions
    
    def _generate_explanation(
        self,
        clause_text: str,
        risk_description: str,
        severity: str,
        guidelines_text: str
    ) -> Tuple[str, str]:
        """
        Generate plain-language explanation and consequences.
        
        Args:
            clause_text: Clause text
            risk_description: Risk description
            severity: Severity level
            guidelines_text: Retrieved legal guidelines
            
        Returns:
            Tuple of (explanation, consequences)
        """
        prompt = f"""You are a legal educator explaining contract risks to non-lawyers.

Clause: "{clause_text}"

Risk: {risk_description}
Severity: {severity}
{guidelines_text}

Provide two things:

1. EXPLANATION (2-3 sentences): Explain in plain language WHY this clause is risky. Avoid legal jargon. If you must use legal terms, define them simply.

2. CONSEQUENCES (2-3 sentences): Explain what could happen if this risk materializes. Be specific about potential impacts.

Format your response as:
EXPLANATION: [your explanation]
CONSEQUENCES: [your consequences]"""
        
        response = self._call_llm_with_retry(prompt)
        
        # Parse response
        explanation = ""
        consequences = ""
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("EXPLANATION:"):
                current_section = "explanation"
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("CONSEQUENCES:"):
                current_section = "consequences"
                consequences = line.replace("CONSEQUENCES:", "").strip()
            elif current_section == "explanation" and line:
                explanation += " " + line
            elif current_section == "consequences" and line:
                consequences += " " + line
        
        # Fallback if parsing fails
        if not explanation:
            explanation = f"This clause presents a {severity} risk: {risk_description}"
        if not consequences:
            consequences = "This could lead to legal or financial complications."
        
        return explanation.strip(), consequences.strip()
    
    def _determine_overall_severity(self, risks: List[Risk]) -> str:
        """
        Determine overall contract severity based on individual risks.
        
        Args:
            risks: List of analyzed risks
            
        Returns:
            Overall severity: "high", "medium", or "low"
        """
        if not risks:
            return "low"
        
        # Count severity levels
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for risk in risks:
            severity_counts[risk.severity] += 1
        
        # Determine overall severity
        if severity_counts["high"] > 0:
            return "high"
        elif severity_counts["medium"] > 0:
            return "medium"
        else:
            return "low"

    
    def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Call LLM with retry logic for malformed responses.
        
        Args:
            prompt: Prompt text
            
        Returns:
            LLM response text
            
        Raises:
            AnalysisError: If all retries fail
        """
        # Use mock LLM if configured
        if self.use_mock:
            return self.mock_llm(prompt)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{self.max_retries}")
                
                # Call Gemini API (old version)
                response = genai.generate_text(
                    prompt=prompt,
                    model=settings.llm_model,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens
                )
                
                # Extract text from response
                if response and response.result:
                    return response.result
                else:
                    raise ValueError("Empty response from LLM")
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    wait_time = settings.retry_delay_seconds * (2 ** attempt)
                    logger.debug(f"Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
        
        # All retries failed
        error_msg = f"LLM call failed after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise AnalysisError(error_msg)
