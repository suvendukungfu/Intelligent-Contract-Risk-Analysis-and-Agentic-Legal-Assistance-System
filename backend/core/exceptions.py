"""Custom exceptions for the Contract Risk Analysis System."""


class ContractAnalysisError(Exception):
    """Base exception for contract analysis errors."""
    pass


class ParseError(ContractAnalysisError):
    """Exception raised when document parsing fails."""
    pass


class ClassificationError(ContractAnalysisError):
    """Exception raised when risk classification fails."""
    pass


class AnalysisError(ContractAnalysisError):
    """Exception raised when agentic analysis fails."""
    pass


class LLMError(ContractAnalysisError):
    """Exception raised when LLM API calls fail."""
    pass


class RAGError(ContractAnalysisError):
    """Exception raised when RAG system fails."""
    pass


class ValidationError(ContractAnalysisError):
    """Exception raised when input validation fails."""
    pass


class TimeoutError(ContractAnalysisError):
    """Exception raised when operations exceed timeout."""
    pass


class FileSizeError(ValidationError):
    """Exception raised when file size exceeds limit."""
    pass


class FileTypeError(ValidationError):
    """Exception raised when file type is not supported."""
    pass
