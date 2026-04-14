"""
Document parser for extracting text from PDF and TXT files.
Implements Requirements 1.1, 1.2, 1.3, 1.4, 1.5.
"""

import re
import logging
from pathlib import Path
from typing import Union, BinaryIO
import chardet
import PyPDF2
import pdfplumber
from fastapi import UploadFile

from backend.api.models import ParsedDocument
from backend.core.exceptions import ParseError, FileTypeError
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentParser:
    """
    Parser for extracting text from contract documents.
    
    Supports:
    - PDF files (using PyPDF2 and pdfplumber)
    - TXT files (with automatic encoding detection)
    
    Features:
    - Whitespace normalization
    - Document structure preservation
    - Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize the document parser."""
        self.supported_extensions = ['.pdf', '.txt']
    
    def parse(self, file: Union[UploadFile, str, Path]) -> ParsedDocument:
        """
        Extract text from uploaded file.
        
        Args:
            file: UploadFile object, file path string, or Path object
            
        Returns:
            ParsedDocument with extracted text and metadata
            
        Raises:
            ParseError: If file cannot be parsed
            FileTypeError: If file type is not supported
        """
        try:
            # Handle different input types
            if isinstance(file, (str, Path)):
                return self._parse_from_path(Path(file))
            else:
                return self._parse_from_upload(file)
                
        except (ParseError, FileTypeError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during parsing: {str(e)}", exc_info=True)
            raise ParseError(
                f"An unexpected error occurred while parsing the document: {str(e)}"
            )
    
    def _parse_from_upload(self, file: UploadFile) -> ParsedDocument:
        """Parse document from FastAPI UploadFile."""
        filename = file.filename or "unknown"
        file_ext = Path(filename).suffix.lower()
        
        # Validate file type
        if file_ext not in self.supported_extensions:
            raise FileTypeError(
                f"Unsupported file type: {file_ext}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )
        
        try:
            # Read file content
            content = file.file.read()
            
            # Check if file is empty
            if not content or len(content) == 0:
                raise ParseError(
                    "The uploaded file is empty. Please upload a file with content."
                )
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text, page_count = self._extract_from_pdf_bytes(content, filename)
            else:  # .txt
                text, page_count = self._extract_from_txt_bytes(content, filename)
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            # Validate extracted text
            if not text or len(text.strip()) < 10:
                raise ParseError(
                    "The document appears to be empty or contains insufficient text. "
                    "Please ensure the file contains readable content."
                )
            
            # Create ParsedDocument
            return ParsedDocument(
                filename=filename,
                text=text,
                page_count=page_count,
                metadata={
                    "file_size": len(content),
                    "file_type": file_ext,
                    "original_length": len(text)
                }
            )
            
        except (ParseError, FileTypeError):
            raise
        except Exception as e:
            logger.error(f"Error parsing {filename}: {str(e)}", exc_info=True)
            raise ParseError(
                f"Failed to parse {filename}. The file may be corrupted, "
                f"password-protected, or in an unsupported format. Error: {str(e)}"
            )
        finally:
            # Reset file pointer for potential reuse
            if hasattr(file.file, 'seek'):
                try:
                    file.file.seek(0)
                except Exception:
                    pass
    
    def _parse_from_path(self, file_path: Path) -> ParsedDocument:
        """Parse document from file path."""
        if not file_path.exists():
            raise ParseError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ParseError(f"Path is not a file: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        # Validate file type
        if file_ext not in self.supported_extensions:
            raise FileTypeError(
                f"Unsupported file type: {file_ext}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check if file is empty
            if not content or len(content) == 0:
                raise ParseError(
                    f"The file {file_path.name} is empty. Please provide a file with content."
                )
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text, page_count = self._extract_from_pdf_bytes(content, file_path.name)
            else:  # .txt
                text, page_count = self._extract_from_txt_bytes(content, file_path.name)
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            # Validate extracted text
            if not text or len(text.strip()) < 10:
                raise ParseError(
                    f"The document {file_path.name} appears to be empty or contains "
                    "insufficient text. Please ensure the file contains readable content."
                )
            
            # Create ParsedDocument
            return ParsedDocument(
                filename=file_path.name,
                text=text,
                page_count=page_count,
                metadata={
                    "file_size": len(content),
                    "file_type": file_ext,
                    "original_length": len(text),
                    "file_path": str(file_path)
                }
            )
            
        except (ParseError, FileTypeError):
            raise
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}", exc_info=True)
            raise ParseError(
                f"Failed to parse {file_path.name}. The file may be corrupted, "
                f"password-protected, or in an unsupported format. Error: {str(e)}"
            )
    
    def _extract_from_pdf_bytes(self, content: bytes, filename: str) -> tuple[str, int]:
        """
        Extract text from PDF file bytes.
        
        Uses pdfplumber as primary method, falls back to PyPDF2 if needed.
        
        Args:
            content: PDF file content as bytes
            filename: Original filename for error messages
            
        Returns:
            Tuple of (extracted_text, page_count)
            
        Raises:
            ParseError: If PDF cannot be parsed
        """
        text_parts = []
        page_count = 0
        
        # Try pdfplumber first (better text extraction)
        try:
            import io
            pdf_file = io.BytesIO(content)
            
            with pdfplumber.open(pdf_file) as pdf:
                page_count = len(pdf.pages)
                
                if page_count == 0:
                    raise ParseError(
                        f"The PDF file {filename} contains no pages. "
                        "Please upload a valid PDF document."
                    )
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                # If pdfplumber extracted text successfully, return it
                if text_parts:
                    return '\n\n'.join(text_parts), page_count
                    
        except Exception as e:
            logger.warning(
                f"pdfplumber failed for {filename}, trying PyPDF2: {str(e)}"
            )
        
        # Fallback to PyPDF2
        try:
            import io
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            page_count = len(pdf_reader.pages)
            
            if page_count == 0:
                raise ParseError(
                    f"The PDF file {filename} contains no pages. "
                    "Please upload a valid PDF document."
                )
            
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            if not text_parts:
                raise ParseError(
                    f"Unable to extract text from {filename}. The PDF may be "
                    "image-based (scanned), password-protected, or corrupted. "
                    "Please try converting it to a text-based PDF or TXT file."
                )
            
            return '\n\n'.join(text_parts), page_count
            
        except PyPDF2.errors.PdfReadError as e:
            raise ParseError(
                f"Failed to read PDF file {filename}. The file may be corrupted, "
                f"password-protected, or in an invalid format. Error: {str(e)}"
            )
        except Exception as e:
            raise ParseError(
                f"An error occurred while extracting text from {filename}: {str(e)}"
            )
    
    def _extract_from_txt_bytes(self, content: bytes, filename: str) -> tuple[str, int]:
        """
        Extract text from TXT file bytes with encoding detection.
        
        Args:
            content: TXT file content as bytes
            filename: Original filename for error messages
            
        Returns:
            Tuple of (extracted_text, page_count=1)
            
        Raises:
            ParseError: If TXT cannot be decoded
        """
        # Detect encoding
        try:
            detection = chardet.detect(content)
            encoding = detection.get('encoding', 'utf-8')
            confidence = detection.get('confidence', 0)
            
            logger.info(
                f"Detected encoding for {filename}: {encoding} "
                f"(confidence: {confidence:.2f})"
            )
            
            # Try detected encoding
            if encoding and confidence > 0.7:
                try:
                    text = content.decode(encoding)
                    return text, 1
                except (UnicodeDecodeError, LookupError):
                    logger.warning(
                        f"Failed to decode {filename} with detected encoding {encoding}"
                    )
            
            # Fallback encodings
            fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for enc in fallback_encodings:
                try:
                    text = content.decode(enc)
                    logger.info(f"Successfully decoded {filename} using {enc}")
                    return text, 1
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # If all encodings fail
            raise ParseError(
                f"Unable to decode text file {filename}. The file encoding is not "
                "supported or the file may be corrupted. Please try saving the file "
                "with UTF-8 encoding."
            )
            
        except Exception as e:
            if isinstance(e, ParseError):
                raise
            raise ParseError(
                f"An error occurred while reading text file {filename}: {str(e)}"
            )
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in extracted text.
        
        - Converts multiple spaces to single space
        - Converts multiple newlines to double newline (paragraph breaks)
        - Removes leading/trailing whitespace
        - Preserves document structure
        
        Args:
            text: Raw extracted text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space (but preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph separator)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove spaces at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from entire document
        text = text.strip()
        
        return text
    
    def validate_file_size(self, file: UploadFile) -> None:
        """
        Validate that file size is within limits.
        
        Args:
            file: UploadFile object
            
        Raises:
            ParseError: If file size exceeds limit
        """
        # Read file to check size
        content = file.file.read()
        file_size = len(content)
        
        # Reset file pointer
        file.file.seek(0)
        
        max_size = settings.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            raise ParseError(
                f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({settings.max_file_size_mb} MB). "
                "Please upload a smaller file."
            )
