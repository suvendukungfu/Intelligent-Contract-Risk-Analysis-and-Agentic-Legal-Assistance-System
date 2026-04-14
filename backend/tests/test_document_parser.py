"""
Unit tests for DocumentParser.
Tests Requirements 1.1, 1.2, 1.3, 1.5.
"""

import pytest
import tempfile
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

from backend.core.document_parser import DocumentParser
from backend.core.exceptions import ParseError, FileTypeError


class TestDocumentParser:
    """Test suite for DocumentParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a DocumentParser instance."""
        return DocumentParser()
    
    def test_parse_valid_txt_file(self, parser):
        """Test parsing a valid text file."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test contract.\n\nClause 1: Test clause.")
            temp_path = Path(f.name)
        
        try:
            # Parse the file
            result = parser.parse(temp_path)
            
            # Assertions
            assert result.filename == temp_path.name
            assert "This is a test contract" in result.text
            assert "Clause 1: Test clause" in result.text
            assert result.page_count == 1
            assert result.metadata['file_type'] == '.txt'
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_parse_empty_file(self, parser):
        """Test parsing an empty file raises ParseError."""
        # Create an empty text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Should raise ParseError
            with pytest.raises(ParseError) as exc_info:
                parser.parse(temp_path)
            
            assert "empty" in str(exc_info.value).lower()
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_parse_unsupported_file_type(self, parser):
        """Test parsing unsupported file type raises FileTypeError."""
        # Create a file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            # Should raise FileTypeError
            with pytest.raises(FileTypeError) as exc_info:
                parser.parse(temp_path)
            
            assert "unsupported" in str(exc_info.value).lower()
            assert ".docx" in str(exc_info.value)
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_whitespace_normalization(self, parser):
        """Test whitespace normalization."""
        # Create a text file with excessive whitespace
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This  has   multiple    spaces.\n\n\n\nAnd multiple newlines.")
            temp_path = Path(f.name)
        
        try:
            # Parse the file
            result = parser.parse(temp_path)
            
            # Check normalization
            assert "multiple    spaces" not in result.text
            assert "This has multiple spaces" in result.text
            assert "\n\n\n" not in result.text
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_whitespace_normalization_idempotence(self, parser):
        """Test that whitespace normalization is idempotent."""
        text = "This  has   multiple    spaces.\n\n\n\nAnd multiple newlines."
        
        # Apply normalization twice
        normalized_once = parser._normalize_whitespace(text)
        normalized_twice = parser._normalize_whitespace(normalized_once)
        
        # Should be identical
        assert normalized_once == normalized_twice
    
    def test_parse_from_upload_file(self, parser):
        """Test parsing from FastAPI UploadFile."""
        # Create a mock UploadFile
        content = b"This is a test contract.\n\nClause 1: Test clause."
        file = UploadFile(
            filename="test_contract.txt",
            file=BytesIO(content)
        )
        
        # Parse the file
        result = parser.parse(file)
        
        # Assertions
        assert result.filename == "test_contract.txt"
        assert "This is a test contract" in result.text
        assert result.page_count == 1
    
    def test_parse_file_with_insufficient_text(self, parser):
        """Test parsing file with very little text raises ParseError."""
        # Create a file with minimal content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hi")  # Less than 10 characters
            temp_path = Path(f.name)
        
        try:
            # Should raise ParseError
            with pytest.raises(ParseError) as exc_info:
                parser.parse(temp_path)
            
            assert "insufficient" in str(exc_info.value).lower()
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_parse_nonexistent_file(self, parser):
        """Test parsing non-existent file raises ParseError."""
        fake_path = Path("/nonexistent/file.txt")
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse(fake_path)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_encoding_detection(self, parser):
        """Test automatic encoding detection for text files."""
        # Create a text file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write UTF-8 encoded text with special characters
            f.write("Contract with special chars: café, naïve, résumé".encode('utf-8'))
            temp_path = Path(f.name)
        
        try:
            # Parse the file
            result = parser.parse(temp_path)
            
            # Check that special characters are preserved
            assert "café" in result.text
            assert "naïve" in result.text
            assert "résumé" in result.text
            
        finally:
            # Clean up
            temp_path.unlink()
