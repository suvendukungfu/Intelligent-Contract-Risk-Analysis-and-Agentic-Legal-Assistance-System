"""
Data models and API schemas for the Contract Risk Analysis System.
Uses Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class ParsedDocument(BaseModel):
    """Represents a parsed contract document."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    text: str
    page_count: int
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "contract.pdf",
                "text": "This is a sample contract...",
                "page_count": 5,
                "upload_timestamp": "2024-01-01T12:00:00",
                "metadata": {"file_size": 1024}
            }
        }


class Clause(BaseModel):
    """Represents a single clause within a contract."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    position: int
    start_char: int = 0
    end_char: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "clause-1",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "The parties agree to...",
                "position": 0,
                "start_char": 0,
                "end_char": 100
            }
        }


class RiskPrediction(BaseModel):
    """Represents a risk prediction for a clause."""
    
    clause_id: str
    risk_label: str = Field(..., pattern="^(high_risk|medium_risk|low_risk|no_risk)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = "v1"
    
    class Config:
        json_schema_extra = {
            "example": {
                "clause_id": "clause-1",
                "risk_label": "high_risk",
                "confidence": 0.87,
                "model_version": "v1"
            }
        }


class Risk(BaseModel):
    """Represents an identified risk in a contract."""
    
    clause_id: str
    clause_text: str
    risk_description: str
    severity: str = Field(..., pattern="^(high|medium|low)$")
    explanation: str
    consequences: str
    mitigation_actions: List[str]
    legal_guidelines: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "clause_id": "clause-1",
                "clause_text": "The parties agree to...",
                "risk_description": "Unlimited liability clause",
                "severity": "high",
                "explanation": "This clause exposes you to unlimited financial risk",
                "consequences": "You could be liable for damages beyond your control",
                "mitigation_actions": ["Add liability cap", "Define scope of liability"],
                "legal_guidelines": ["Indian Contract Act Section 73"]
            }
        }


class RiskReport(BaseModel):
    """Represents a comprehensive risk analysis report."""
    
    contract_summary: str
    identified_risks: List[Risk]
    overall_severity: str = Field(..., pattern="^(high|medium|low)$")
    legal_disclaimer: str = (
        "This analysis is provided for informational purposes only and does not "
        "constitute legal advice. Please consult with a qualified attorney for "
        "legal guidance specific to your situation."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "contract_summary": "This is a standard employment agreement...",
                "identified_risks": [],
                "overall_severity": "medium",
                "legal_disclaimer": "This analysis is provided for informational purposes only..."
            }
        }


class LegalGuideline(BaseModel):
    """Represents a legal guideline retrieved from RAG system."""
    
    text: str
    source: str
    url: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Under Indian Contract Act...",
                "source": "Indian Kanoon",
                "url": "https://indiankanoon.org/...",
                "relevance_score": 0.92
            }
        }


class TrainingExample(BaseModel):
    """Represents a training data example."""
    
    clause_text: str
    risk_label: str = Field(..., pattern="^(high_risk|medium_risk|low_risk|no_risk)$")
    contract_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "clause_text": "The employee agrees to unlimited overtime...",
                "risk_label": "high_risk",
                "contract_type": "employment",
                "jurisdiction": "India"
            }
        }


# API Request/Response Models

class AnalysisRequest(BaseModel):
    """Request model for contract analysis."""
    
    milestone: int = Field(..., ge=1, le=2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "milestone": 1
            }
        }


class Milestone1Response(BaseModel):
    """Response model for Milestone 1 analysis."""
    
    document_id: str
    clauses: List[Dict[str, Any]]
    summary: Dict[str, int]
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "clauses": [
                    {
                        "id": "clause-1",
                        "text": "...",
                        "risk_label": "high_risk",
                        "confidence": 0.87,
                        "position": 0
                    }
                ],
                "summary": {
                    "high_risk": 2,
                    "medium_risk": 5,
                    "low_risk": 10,
                    "no_risk": 15
                }
            }
        }


class Milestone2Response(BaseModel):
    """Response model for Milestone 2 analysis."""
    
    document_id: str
    report: RiskReport
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "report": {
                    "contract_summary": "...",
                    "identified_risks": [],
                    "overall_severity": "medium",
                    "legal_disclaimer": "..."
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "PARSE_ERROR",
                    "message": "Unable to extract text from PDF file",
                    "details": "PDF appears to be corrupted or password-protected",
                    "suggestion": "Please try uploading a different file or convert to text format"
                }
            }
        }
