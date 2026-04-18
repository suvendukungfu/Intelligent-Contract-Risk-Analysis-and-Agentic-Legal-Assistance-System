import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.models import ParsedDocument, Clause, RiskPrediction
from core.document_parser import DocumentParser
from core.clause_segmenter import ClauseSegmenter
from core.feature_extractor import get_feature_extractor
from core.risk_classifier import get_risk_classifier
from core.rag_system import RAGSystem
from core.agentic_assistant import AgenticAssistant
from core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_milestone2():
    settings = get_settings()
    
    logger.info("Testing Milestone 2 analysis pipeline")
    
    test_file = Path(__file__).parent / "data" / "test_nda.txt"
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"Using test file: {test_file}")
    
    with open(test_file, 'rb') as f:
        from io.BytesIO
        from fastapi import UploadFile
        
        content = f.read()
        file_obj = BytesIO(content)
        upload_file = UploadFile(filename="test_nda.txt", file=file_obj)
        
        logger.info("Step 1: Parsing document")
        parser = DocumentParser()
        document = parser.parse(upload_file)
        logger.info(f"Document parsed: {document.id}, {document.page_count} pages")
        
        logger.info("Step 2: Segmenting clauses")
        segmenter = ClauseSegmenter()
        clauses = segmenter.segment(document)
        logger.info(f"Segmented into {len(clauses)} clauses")
        
        logger.info("Step 3: Extracting features")
        feature_extractor = get_feature_extractor()
        features = feature_extractor.extract(clauses)
        logger.info(f"Features extracted: {features.shape}")
        
        logger.info("Step 4: Classifying clauses")
        model_path = Path(settings.model_path) / settings.classifier_model
        classifier = get_risk_classifier(str(model_path))
        predictions = classifier.predict(features)
        
        for clause, pred in zip(clauses, predictions):
            pred.clause_id = clause.id
        
        logger.info(f"Classification complete: {len(predictions)} predictions")
        
        logger.info("Step 5: Agentic AI analysis")
        rag = RAGSystem(vector_store_path=settings.vector_store_path)
        assistant = AgenticAssistant(rag_system=rag)
        
        report = assistant.analyze(
            contract=document,
            clauses=clauses,
            ml_predictions=predictions
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"\nContract Summary:\n{report.contract_summary}")
        logger.info(f"\nOverall Severity: {report.overall_severity.upper()}")
        logger.info(f"\nIdentified Risks: {len(report.identified_risks)}")
        
        for i, risk in enumerate(report.identified_risks, 1):
            logger.info(f"\n--- Risk {i} ---")
            logger.info(f"Clause: {risk.clause_text[:100]}...")
            logger.info(f"Risk: {risk.risk_description}")
            logger.info(f"Severity: {risk.severity}")
            logger.info(f"Explanation: {risk.explanation}")
            logger.info(f"Mitigation Actions: {len(risk.mitigation_actions)}")
            logger.info(f"Legal Guidelines: {len(risk.legal_guidelines)}")
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST PASSED")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    test_milestone2()
