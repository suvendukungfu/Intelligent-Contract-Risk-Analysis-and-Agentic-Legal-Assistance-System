"""Report Generator for Contract Risk Analysis System."""

import logging
from typing import Dict, List, Any, Optional

from api.models import RiskReport, Risk, ParsedDocument, Clause, RiskPrediction

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured risk reports from analysis results."""
    
    def __init__(self):
        self.default_disclaimer = (
            "This analysis is provided for informational purposes only and does not "
            "constitute legal advice. Please consult with a qualified attorney for "
            "legal guidance specific to your situation."
        )
    
    def generate(
        self,
        contract_summary: str,
        identified_risks: List[Dict[str, Any]],
        overall_severity: Optional[str] = None,
        legal_disclaimer: Optional[str] = None
    ) -> RiskReport:
        """Generate a structured risk report from analysis results."""
        logger.info("Generating risk report...")
        
        contract_summary = self._validate_summary(contract_summary)
        identified_risks = self._validate_risks(identified_risks)
        overall_severity = self._determine_overall_severity(identified_risks, overall_severity)
        legal_disclaimer = legal_disclaimer or self.default_disclaimer
        
        risk_objects = []
        for risk_data in identified_risks:
            try:
                risk = Risk(
                    clause_id=risk_data.get('clause_id', 'unknown'),
                    clause_text=risk_data.get('clause_text', ''),
                    risk_description=risk_data.get('risk_description', ''),
                    severity=risk_data.get('severity', 'medium'),
                    explanation=risk_data.get('explanation', ''),
                    consequences=risk_data.get('consequences', ''),
                    mitigation_actions=risk_data.get('mitigation_actions', []),
                    legal_guidelines=risk_data.get('legal_guidelines', [])
                )
                risk_objects.append(risk)
            except Exception as e:
                logger.warning(f"Skipping invalid risk entry: {e}")
                continue
        
        report = RiskReport(
            contract_summary=contract_summary,
            identified_risks=risk_objects,
            overall_severity=overall_severity,
            legal_disclaimer=legal_disclaimer
        )
        
        logger.info(f"Report generated: {len(risk_objects)} risks, severity: {overall_severity}")
        return report
    
    def generate_from_agentic_analysis(self, analysis_results: Dict[str, Any]) -> RiskReport:
        """Generate report from raw agentic assistant output."""
        return self.generate(
            contract_summary=analysis_results.get('contract_summary', ''),
            identified_risks=analysis_results.get('risks', []),
            overall_severity=analysis_results.get('overall_severity'),
            legal_disclaimer=analysis_results.get('legal_disclaimer')
        )
    
    def generate_from_ml_predictions(
        self,
        document: ParsedDocument,
        clauses: List[Clause],
        predictions: List[RiskPrediction],
        include_explanations: bool = False
    ) -> RiskReport:
        """Generate a basic report from ML predictions only."""
        logger.info("Generating report from ML predictions...")
        
        contract_summary = self._generate_basic_summary(document, clauses, predictions)
        
        identified_risks = []
        for clause, prediction in zip(clauses, predictions):
            if prediction.risk_label in ['high_risk', 'medium_risk']:
                severity = prediction.risk_label.replace('_risk', '')
                
                risk_data = {
                    'clause_id': clause.id,
                    'clause_text': clause.text,
                    'risk_description': f"{severity.capitalize()} risk clause identified",
                    'severity': severity,
                    'explanation': self._generate_basic_explanation(clause.text, severity) if include_explanations else "ML-based risk classification",
                    'consequences': "Potential legal or financial implications",
                    'mitigation_actions': ["Review with legal counsel"],
                    'legal_guidelines': []
                }
                identified_risks.append(risk_data)
        
        overall_severity = self._determine_overall_severity(identified_risks)
        
        return self.generate(
            contract_summary=contract_summary,
            identified_risks=identified_risks,
            overall_severity=overall_severity
        )
    
    def _validate_summary(self, summary: str) -> str:
        """Validate and clean contract summary."""
        if not summary or not summary.strip():
            logger.warning("Empty contract summary provided")
            return "No summary available."
        
        summary = summary.strip()
        if len(summary) > 5000:
            logger.warning(f"Summary too long ({len(summary)} chars), truncating")
            summary = summary[:5000] + "..."
        
        return summary
    
    def _validate_risks(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean risk entries."""
        if not risks:
            logger.info("No risks provided")
            return []
        
        validated_risks = []
        for i, risk in enumerate(risks):
            if not isinstance(risk, dict):
                logger.warning(f"Risk {i} is not a dictionary, skipping")
                continue
            
            severity = risk.get('severity', 'medium').lower()
            if severity not in ['high', 'medium', 'low']:
                logger.warning(f"Invalid severity '{severity}' in risk {i}, defaulting to 'medium'")
                risk['severity'] = 'medium'
            else:
                risk['severity'] = severity
            
            if 'mitigation_actions' in risk and not isinstance(risk['mitigation_actions'], list):
                risk['mitigation_actions'] = [str(risk['mitigation_actions'])]
            
            if 'legal_guidelines' in risk and not isinstance(risk['legal_guidelines'], list):
                risk['legal_guidelines'] = [str(risk['legal_guidelines'])]
            
            for field in ['clause_text', 'risk_description', 'explanation', 'consequences']:
                if field in risk and not isinstance(risk[field], str):
                    risk[field] = str(risk[field])
            
            validated_risks.append(risk)
        
        logger.info(f"Validated {len(validated_risks)} risks")
        return validated_risks
    
    def _determine_overall_severity(
        self,
        risks: List[Dict[str, Any]],
        provided_severity: Optional[str] = None
    ) -> str:
        """Determine overall severity based on identified risks."""
        if provided_severity:
            severity = provided_severity.lower()
            if severity in ['high', 'medium', 'low']:
                return severity
            logger.warning(f"Invalid provided severity '{provided_severity}', calculating from risks")
        
        if not risks:
            return 'low'
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for risk in risks:
            severity = risk.get('severity', 'medium').lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        if severity_counts['high'] > 0:
            return 'high'
        elif severity_counts['medium'] > 0:
            return 'medium'
        else:
            return 'low'
    
    def _generate_basic_summary(
        self,
        document: ParsedDocument,
        clauses: List[Clause],
        predictions: List[RiskPrediction]
    ) -> str:
        """Generate a basic contract summary from document metadata."""
        risk_counts = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0, 'no_risk': 0}
        for pred in predictions:
            if pred.risk_label in risk_counts:
                risk_counts[pred.risk_label] += 1
        
        summary = (
            f"This contract contains {len(clauses)} clauses across {document.page_count} page(s). "
            f"ML analysis identified {risk_counts['high_risk']} high-risk clauses, "
            f"{risk_counts['medium_risk']} medium-risk clauses, "
            f"{risk_counts['low_risk']} low-risk clauses, and "
            f"{risk_counts['no_risk']} clauses with no significant risk."
        )
        
        return summary
    
    def _generate_basic_explanation(self, clause_text: str, severity: str) -> str:
        """Generate a basic explanation for a risky clause."""
        explanations = {
            'high': "This clause has been identified as high risk by our ML model. It may contain terms that could expose you to significant legal or financial liability. We strongly recommend reviewing this clause with legal counsel.",
            'medium': "This clause has been identified as medium risk by our ML model. It may contain terms that warrant careful review and consideration. Consider consulting with legal counsel for clarification.",
            'low': "This clause has been identified as low risk by our ML model. However, all contract terms should be reviewed carefully."
        }
        return explanations.get(severity, explanations['low'])
    
    def format_report_as_text(self, report: RiskReport) -> str:
        """Format a RiskReport as plain text for display or export."""
        lines = [
            "=" * 80,
            "CONTRACT RISK ANALYSIS REPORT",
            "=" * 80,
            "",
            "CONTRACT SUMMARY",
            "-" * 80,
            report.contract_summary,
            "",
            "OVERALL RISK SEVERITY",
            "-" * 80,
            f"{report.overall_severity.upper()}",
            "",
            f"IDENTIFIED RISKS ({len(report.identified_risks)})",
            "-" * 80
        ]
        
        for i, risk in enumerate(report.identified_risks, 1):
            lines.extend([
                f"\nRisk #{i}: {risk.risk_description}",
                f"Severity: {risk.severity.upper()}",
                f"\nClause: \"{risk.clause_text}\"",
                f"\nExplanation: {risk.explanation}",
                f"\nConsequences: {risk.consequences}"
            ])
            
            if risk.mitigation_actions:
                lines.append("\nMitigation Actions:")
                lines.extend([f"  • {action}" for action in risk.mitigation_actions])
            
            if risk.legal_guidelines:
                lines.append("\nLegal Guidelines:")
                lines.extend([f"  • {guideline}" for guideline in risk.legal_guidelines])
            
            lines.append("")
        
        lines.extend([
            "LEGAL DISCLAIMER",
            "-" * 80,
            report.legal_disclaimer,
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)

_report_generator_instance: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create a singleton ReportGenerator instance."""
    global _report_generator_instance
    if _report_generator_instance is None:
        _report_generator_instance = ReportGenerator()
    return _report_generator_instance
