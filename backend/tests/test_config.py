"""Tests for configuration management."""

import os
import pytest
from pydantic import ValidationError

from core.config import Settings


def test_settings_default_values():
    """Test that settings load with default values."""
    settings = Settings()
    
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.max_file_size_mb == 10
    assert settings.analysis_timeout_seconds == 60
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_settings_cors_origins_parsing():
    """Test that CORS origins are parsed correctly."""
    settings = Settings(cors_origins="http://localhost:3000,http://localhost:5173")
    
    assert isinstance(settings.cors_origins, list)
    assert len(settings.cors_origins) == 2
    assert "http://localhost:3000" in settings.cors_origins
    assert "http://localhost:5173" in settings.cors_origins


def test_settings_allowed_extensions_parsing():
    """Test that allowed extensions are parsed correctly."""
    settings = Settings(allowed_extensions="pdf,txt,docx")
    
    assert isinstance(settings.allowed_extensions, list)
    assert len(settings.allowed_extensions) == 3
    assert "pdf" in settings.allowed_extensions
    assert "txt" in settings.allowed_extensions
    assert "docx" in settings.allowed_extensions


def test_settings_invalid_max_file_size():
    """Test that negative file size raises validation error."""
    with pytest.raises(ValidationError):
        Settings(max_file_size_mb=-1)


def test_settings_invalid_timeout():
    """Test that negative timeout raises validation error."""
    with pytest.raises(ValidationError):
        Settings(analysis_timeout_seconds=-1)


def test_settings_gemini_api_key_required():
    """Test that Gemini API key is required when using Gemini provider."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(llm_provider="gemini", gemini_api_key=None)
    
    assert "GEMINI_API_KEY must be set" in str(exc_info.value)


def test_settings_gemini_api_key_provided():
    """Test that settings load successfully with Gemini API key."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test_api_key_123"
    )
    
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key == "test_api_key_123"


def test_settings_from_env_file(tmp_path, monkeypatch):
    """Test that settings load from .env file."""
    # Create temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_ENV=production\n"
        "LOG_LEVEL=DEBUG\n"
        "MAX_FILE_SIZE_MB=20\n"
        "GEMINI_API_KEY=test_key\n"
    )
    
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    settings = Settings()
    
    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"
    assert settings.max_file_size_mb == 20
    assert settings.gemini_api_key == "test_key"
