"""Tests for validation utilities."""

import pytest
from fastapi import UploadFile
from io import BytesIO

from core.validators import (
    validate_file_upload,
    validate_file_content,
    get_file_extension
)
from core.exceptions import FileSizeError, FileTypeError


def test_validate_file_upload_valid_pdf():
    """Test validation passes for valid PDF file."""
    file = UploadFile(
        filename="contract.pdf",
        file=BytesIO(b"test content")
    )
    
    # Should not raise exception
    validate_file_upload(file)


def test_validate_file_upload_valid_txt():
    """Test validation passes for valid TXT file."""
    file = UploadFile(
        filename="contract.txt",
        file=BytesIO(b"test content")
    )
    
    # Should not raise exception
    validate_file_upload(file)


def test_validate_file_upload_invalid_extension():
    """Test validation fails for unsupported file type."""
    file = UploadFile(
        filename="contract.docx",
        file=BytesIO(b"test content")
    )
    
    with pytest.raises(FileTypeError) as exc_info:
        validate_file_upload(file)
    
    assert "not supported" in str(exc_info.value)


def test_validate_file_upload_no_filename():
    """Test validation fails when filename is missing."""
    file = UploadFile(
        filename=None,
        file=BytesIO(b"test content")
    )
    
    with pytest.raises(FileTypeError) as exc_info:
        validate_file_upload(file)
    
    assert "Filename is required" in str(exc_info.value)


def test_validate_file_content_valid():
    """Test content validation passes for valid content."""
    content = b"This is a valid contract with enough content."
    
    # Should not raise exception
    validate_file_content(content)


def test_validate_file_content_empty():
    """Test content validation fails for empty content."""
    content = b""
    
    with pytest.raises(FileSizeError) as exc_info:
        validate_file_content(content)
    
    assert "empty" in str(exc_info.value).lower()


def test_validate_file_content_too_small():
    """Test content validation fails for content that's too small."""
    content = b"tiny"
    
    with pytest.raises(FileSizeError) as exc_info:
        validate_file_content(content)
    
    assert "too small" in str(exc_info.value).lower()


def test_validate_file_content_too_large():
    """Test content validation fails for content exceeding size limit."""
    # Create content larger than 10MB
    content = b"x" * (11 * 1024 * 1024)
    
    with pytest.raises(FileSizeError) as exc_info:
        validate_file_content(content)
    
    assert "exceeds maximum" in str(exc_info.value)


def test_get_file_extension_pdf():
    """Test file extension extraction for PDF."""
    assert get_file_extension("contract.pdf") == "pdf"


def test_get_file_extension_txt():
    """Test file extension extraction for TXT."""
    assert get_file_extension("document.txt") == "txt"


def test_get_file_extension_uppercase():
    """Test file extension extraction handles uppercase."""
    assert get_file_extension("CONTRACT.PDF") == "pdf"


def test_get_file_extension_multiple_dots():
    """Test file extension extraction with multiple dots in filename."""
    assert get_file_extension("my.contract.v2.pdf") == "pdf"


def test_get_file_extension_no_extension():
    """Test file extension extraction with no extension."""
    assert get_file_extension("contract") == ""
