"""
Configuration management for the Contract Risk Analysis System.
Loads environment variables and provides typed configuration objects.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    analysis_timeout_seconds: int = Field(default=60, env="ANALYSIS_TIMEOUT_SECONDS")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS"
    )
    
    # ML Models Configuration
    model_path: str = Field(default="./ml/models", env="MODEL_PATH")
    classifier_model: str = Field(default="risk_classifier_v1.pkl", env="CLASSIFIER_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # LLM Configuration
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    llm_model: str = Field(default="gemini-1.5-flash", env="LLM_MODEL")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    
    # RAG Configuration
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    vector_store_type: str = Field(default="chromadb", env="VECTOR_STORE_TYPE")
    rag_top_k: int = Field(default=3, env="RAG_TOP_K")
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # File Upload Configuration
    allowed_extensions: str = Field(default="pdf,txt", env="ALLOWED_EXTENSIONS")
    max_upload_size_bytes: int = Field(default=10485760, env="MAX_UPLOAD_SIZE_BYTES")
    
    # Retry Configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay_seconds: int = Field(default=1, env="RETRY_DELAY_SECONDS")
    
    # Database (optional)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Monitoring (optional)
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    @validator("cors_origins")
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in v.split(",") if origin.strip()]
    
    @validator("allowed_extensions")
    def parse_allowed_extensions(cls, v: str) -> List[str]:
        """Parse comma-separated file extensions into a list."""
        return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
    
    @validator("max_file_size_mb")
    def validate_max_file_size(cls, v: int) -> int:
        """Ensure max file size is positive."""
        if v <= 0:
            raise ValueError("max_file_size_mb must be positive")
        return v
    
    @validator("analysis_timeout_seconds")
    def validate_timeout(cls, v: int) -> int:
        """Ensure timeout is positive."""
        if v <= 0:
            raise ValueError("analysis_timeout_seconds must be positive")
        return v
    
    @validator("gemini_api_key")
    def validate_api_key(cls, v: Optional[str], values: dict) -> Optional[str]:
        """Validate that API key is provided for Gemini provider."""
        llm_provider = values.get("llm_provider", "").lower()
        if llm_provider == "gemini" and not v:
            raise ValueError(
                "GEMINI_API_KEY must be set when using Gemini as LLM provider"
            )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
