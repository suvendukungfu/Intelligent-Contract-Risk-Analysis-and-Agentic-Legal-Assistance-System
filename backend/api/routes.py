"""
API routes for the Contract Risk Analysis System.
Implements endpoints for Milestone 1 and Milestone 2 analysis.
"""

import logging
from typing import Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from api.models import Milestone1Response, Milestone2Response, ErrorResponse
from core.document_parser import DocumentParser
from core.clause_segmenter import ClauseSegmenter
from core.feature_extractor import get_feature_extractor
from core.risk_classifier import get_risk_classifier
from core.rag_system import RAGSystem
from core.agentic_assistant import AgenticAssistant, AnalysisError
from core.exceptions import (
    ParseError, 
    FileTypeError, 
    FileSizeError,
    ClassificationError
)
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create API router
router = APIRouter()

# Initialize components (lazy loading)
_document_parser = None
_clause_segmenter = None
_feature_extractor = None
_risk_classifier = None
_rag_system = None
_agentic_assistant = None


def get_document_parser() -> DocumentParser:
    """Get or create DocumentParser instance."""
    global _document_parser
    if _document_parser is None:
        _document_parser = DocumentParser()
    return _document_parser


def get_clause_segmenter() -> ClauseSegmenter:
    """Get or create ClauseSegmenter instance."""
    global _clause_segmenter
    if _clause_segmenter is None:
        _clause_segmenter = ClauseSegmenter()
    return _clause_segmenter


def get_rag_system() -> RAGSystem:
    """Get or create RAGSystem instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(vector_store_path=settings.vector_store_path)
    return _rag_system


def get_agentic_assistant() -> AgenticAssistant:
    """Get or create AgenticAssistant instance."""
    global _agentic_assistant
    if _agentic_assistant is None:
        rag_system = get_rag_system()
        _agentic_assistant = AgenticAssistant(rag_system=rag_system)
    return _agentic_assistant


def _validate_file_type(filename: str) -> None:
    """
    Validate that file type is supported.
    
    Args:
        filename: Name of uploaded file
        
    Raises:
        HTTPException: If file type is not supported
    """
    file_ext = Path(filename).suffix.lower()
    allowed_extensions = ['.pdf', '.txt']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": f"Unsupported file type: {file_ext}",
                    "details": f"Only PDF and TXT files are supported",
                    "suggestion": "Please upload a PDF or TXT file"
                }
            }
        )


def _validate_file_size(file: UploadFile) -> None:
    """
    Validate that file size is within limits.
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file size exceeds limit
    """
    # Read file to check size
    content = file.file.read()
    file_size = len(content)
    
    # Reset file pointer
    file.file.seek(0)
    
    max_size = settings.max_file_size_mb * 1024 * 1024
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"File size exceeds maximum allowed size",
                    "details": f"File size: {file_size / 1024 / 1024:.2f} MB, "
                              f"Maximum: {settings.max_file_size_mb} MB",
                    "suggestion": f"Please upload a file smaller than {settings.max_file_size_mb} MB"
                }
            }
        )


def _calculate_summary(predictions: list) -> Dict[str, int]:
    """
    Calculate summary counts by risk level.
    
    Args:
        predictions: List of RiskPrediction objects
        
    Returns:
        Dictionary with counts for each risk level
    """
    summary = {
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "no_risk": 0
    }
    
    for pred in predictions:
        risk_label = pred.risk_label
        if risk_label in summary:
            summary[risk_label] += 1
    
    return summary


@router.post(
    "/analyze/milestone1",
    response_model=Milestone1Response,
    status_code=status.HTTP_200_OK,
    summary="Analyze contract with ML-based risk classification",
    description="Upload a contract document (PDF or TXT) and receive ML-based risk "
                "classification for each clause with confidence scores."
)
async def analyze_milestone1(
    file: UploadFile = File(..., description="Contract document (PDF or TXT)")
) -> Milestone1Response:
    """
    Milestone 1: ML-Based Risk Classification
    
    This endpoint processes a contract document and returns:
    - Segmented clauses with risk labels
    - Confidence scores for each prediction
    - Summary counts by risk level
    
    Requirements: 1.1, 1.2, 12.1, 12.2
    """
    logger.info(f"Received Milestone 1 analysis request for file: {file.filename}")
    
    try:
        # Step 1: Validate file type
        _validate_file_type(file.filename)
        logger.debug(f"File type validation passed for {file.filename}")
        
        # Step 2: Validate file size
        _validate_file_size(file)
        logger.debug(f"File size validation passed for {file.filename}")
        
        # Step 3: Parse document
        parser = get_document_parser()
        try:
            document = parser.parse(file)
            logger.info(
                f"Document parsed successfully: {document.id}, "
                f"pages: {document.page_count}, "
                f"text length: {len(document.text)}"
            )
        except (ParseError, FileTypeError) as e:
            logger.error(f"Parse error for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "PARSE_ERROR",
                        "message": "Unable to extract text from document",
                        "details": str(e),
                        "suggestion": "Please ensure the file is not corrupted, "
                                    "password-protected, or in an unsupported format"
                    }
                }
            )
        
        # Step 4: Segment clauses
        segmenter = get_clause_segmenter()
        clauses = segmenter.segment(document)
        logger.info(f"Document segmented into {len(clauses)} clauses")
        
        if not clauses:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "SEGMENTATION_ERROR",
                        "message": "Unable to segment document into clauses",
                        "details": "The document structure could not be analyzed",
                        "suggestion": "Please ensure the document contains readable text"
                    }
                }
            )
        
        # Step 5: Extract features
        feature_extractor = get_feature_extractor(
            model_name=settings.embedding_model
        )
        try:
            features = feature_extractor.extract(clauses)
            logger.info(f"Features extracted: shape {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "FEATURE_EXTRACTION_ERROR",
                        "message": "Failed to extract features from clauses",
                        "details": str(e),
                        "suggestion": "Please try again or contact support"
                    }
                }
            )
        
        # Step 6: Classify clauses
        model_path = Path(settings.model_path) / settings.classifier_model
        classifier = get_risk_classifier(
            model_path=str(model_path) if model_path.exists() else None
        )
        
        if not classifier.is_trained:
            logger.error("Risk classifier is not trained")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "MODEL_NOT_READY",
                        "message": "Risk classification model is not available",
                        "details": "The ML model has not been trained yet",
                        "suggestion": "Please contact the administrator to train the model"
                    }
                }
            )
        
        try:
            predictions = classifier.predict(features)
            logger.info(f"Classification complete: {len(predictions)} predictions")
            
            # Update clause IDs in predictions
            for clause, prediction in zip(clauses, predictions):
                prediction.clause_id = clause.id
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "CLASSIFICATION_ERROR",
                        "message": "Failed to classify clauses",
                        "details": str(e),
                        "suggestion": "Please try again or contact support"
                    }
                }
            )
        
        # Step 7: Format response
        clauses_data = []
        for clause, prediction in zip(clauses, predictions):
            clauses_data.append({
                "id": clause.id,
                "text": clause.text,
                "risk_label": prediction.risk_label,
                "confidence": prediction.confidence,
                "position": clause.position
            })
        
        # Calculate summary
        summary = _calculate_summary(predictions)
        
        response = Milestone1Response(
            document_id=document.id,
            clauses=clauses_data,
            summary=summary
        )
        
        logger.info(
            f"Milestone 1 analysis complete for {file.filename}: "
            f"{len(clauses)} clauses, summary: {summary}"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            f"Unexpected error during Milestone 1 analysis: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during analysis",
                    "details": str(e),
                    "suggestion": "Please try again or contact support"
                }
            }
        )


@router.post(
    "/analyze/milestone2",
    response_model=Milestone2Response,
    status_code=status.HTTP_200_OK,
    summary="Analyze contract with Agentic AI",
    description="Upload a contract document (PDF or TXT) and receive comprehensive AI-powered "
                "risk analysis with legal guidelines, severity assessment, and mitigation actions."
)
async def analyze_milestone2(
    file: UploadFile = File(..., description="Contract document (PDF or TXT)")
) -> Milestone2Response:
    """
    Milestone 2: Agentic AI Analysis
    
    This endpoint processes a contract document and returns:
    - Contract summary
    - Identified risks with detailed analysis
    - Legal guidelines and citations
    - Severity assessment
    - Mitigation actions
    - Plain-language explanations
    
    Requirements: 2.1, 2.2, 2.3, 12.3, 12.4
    """
    logger.info(f"Received Milestone 2 analysis request for file: {file.filename}")
    
    try:
        # Step 1: Validate file type
        _validate_file_type(file.filename)
        logger.debug(f"File type validation passed for {file.filename}")
        
        # Step 2: Validate file size
        _validate_file_size(file)
        logger.debug(f"File size validation passed for {file.filename}")
        
        # Step 3: Parse document
        parser = get_document_parser()
        try:
            document = parser.parse(file)
            logger.info(
                f"Document parsed successfully: {document.id}, "
                f"pages: {document.page_count}, "
                f"text length: {len(document.text)}"
            )
        except (ParseError, FileTypeError) as e:
            logger.error(f"Parse error for {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "PARSE_ERROR",
                        "message": "Unable to extract text from document",
                        "details": str(e),
                        "suggestion": "Please ensure the file is not corrupted, "
                                    "password-protected, or in an unsupported format"
                    }
                }
            )
        
        # Step 4: Segment clauses
        segmenter = get_clause_segmenter()
        clauses = segmenter.segment(document)
        logger.info(f"Document segmented into {len(clauses)} clauses")
        
        if not clauses:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "SEGMENTATION_ERROR",
                        "message": "Unable to segment document into clauses",
                        "details": "The document structure could not be analyzed",
                        "suggestion": "Please ensure the document contains readable text"
                    }
                }
            )
        
        # Step 5: Extract features
        feature_extractor = get_feature_extractor(
            model_name=settings.embedding_model
        )
        try:
            features = feature_extractor.extract(clauses)
            logger.info(f"Features extracted: shape {features.shape}")
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "FEATURE_EXTRACTION_ERROR",
                        "message": "Failed to extract features from clauses",
                        "details": str(e),
                        "suggestion": "Please try again or contact support"
                    }
                }
            )
        
        # Step 6: Classify clauses (ML predictions as hints for AI)
        model_path = Path(settings.model_path) / settings.classifier_model
        classifier = get_risk_classifier(
            model_path=str(model_path) if model_path.exists() else None
        )
        
        if not classifier.is_trained:
            logger.error("Risk classifier is not trained")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "MODEL_NOT_READY",
                        "message": "Risk classification model is not available",
                        "details": "The ML model has not been trained yet",
                        "suggestion": "Please contact the administrator to train the model"
                    }
                }
            )
        
        try:
            predictions = classifier.predict(features)
            logger.info(f"Classification complete: {len(predictions)} predictions")
            
            for clause, prediction in zip(clauses, predictions):
                prediction.clause_id = clause.id
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "CLASSIFICATION_ERROR",
                        "message": "Failed to classify clauses",
                        "details": str(e),
                        "suggestion": "Please try again or contact support"
                    }
                }
            )
        
        # Step 7: Agentic AI Analysis
        logger.info("Starting agentic AI analysis")
        assistant = get_agentic_assistant()
        
        try:
            report = assistant.analyze(
                contract=document,
                clauses=clauses,
                ml_predictions=predictions
            )
            logger.info(
                f"Agentic analysis complete: {len(report.identified_risks)} risks identified, "
                f"overall severity: {report.overall_severity}"
            )
        except AnalysisError as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "ANALYSIS_ERROR",
                        "message": "Failed to complete AI analysis",
                        "details": str(e),
                        "suggestion": "The analysis may have timed out. Please try again with a shorter document"
                    }
                }
            )
        except Exception as e:
            logger.error(f"Unexpected analysis error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "code": "ANALYSIS_ERROR",
                        "message": "An unexpected error occurred during analysis",
                        "details": str(e),
                        "suggestion": "Please try again or contact support"
                    }
                }
            )
        
        # Step 8: Format response
        response = Milestone2Response(
            document_id=document.id,
            report=report
        )
        
        logger.info(
            f"Milestone 2 analysis complete for {file.filename}: "
            f"{len(report.identified_risks)} risks, severity: {report.overall_severity}"
        )
        
        return response
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            f"Unexpected error during Milestone 2 analysis: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during analysis",
                    "details": str(e),
                    "suggestion": "Please try again or contact support"
                }
            }
        )


@router.get(
    "/health",
    summary="Health check for API routes",
    description="Check if the API routes are operational"
)
async def health_check() -> Dict[str, str]:
    """Health check endpoint for API routes."""
    return {
        "status": "healthy",
        "service": "api-routes"
    }
