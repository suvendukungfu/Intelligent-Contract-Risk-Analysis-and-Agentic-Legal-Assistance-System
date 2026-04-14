"""Validation utilities for file uploads and inputs."""

import os
from typing import BinaryIO
from fastapi import UploadFile

from .config import settings
from .exceptions import FileSizeError, FileTypeError


def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file meets requirements.
    
    Args:
        file: Uploaded file object
        
    Raises:
        FileTypeError: If file extension is not allowed
        FileSizeError: If file size exceeds limit
    """
    # Validate file extension
    if not file.filename:
        raise FileTypeError("Filename is required")
    
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if file_ext not in settings.allowed_extensions:
        raise FileTypeError(
            f"File type '.{file_ext}' not supported. "
            f"Allowed types: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate file size (if file object has size attribute)
    if hasattr(file, 'size') and file.size:
        if file.size > settings.max_upload_size_bytes:
            max_size_mb = settings.max_upload_size_bytes / (1024 * 1024)
            raise FileSizeError(
                f"File size exceeds maximum allowed size of {max_size_mb:.1f}MB"
            )


def validate_file_content(content: bytes) -> None:
    """
    Validate file content is not empty and within size limits.
    
    Args:
        content: File content as bytes
        
    Raises:
        FileSizeError: If content is empty or exceeds limit
    """
    if not content:
        raise FileSizeError("File is empty")
    
    if len(content) > settings.max_upload_size_bytes:
        max_size_mb = settings.max_upload_size_bytes / (1024 * 1024)
        raise FileSizeError(
            f"File size exceeds maximum allowed size of {max_size_mb:.1f}MB"
        )
    
    # Minimum file size check (at least 10 characters)
    if len(content) < 10:
        raise FileSizeError("File is too small (minimum 10 bytes)")


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension without the dot (lowercase)
    """
    return os.path.splitext(filename)[1].lower().lstrip('.')
