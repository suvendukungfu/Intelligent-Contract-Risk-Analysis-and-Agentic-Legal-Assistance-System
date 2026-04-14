"""
Main application entry point for Contract Risk Analysis System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from core.config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Contract Risk Analysis System",
    description="Intelligent Contract Risk Analysis and Agentic Legal Assistance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting Contract Risk Analysis System in {settings.app_env} mode")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Max file size: {settings.max_file_size_mb}MB")
    logger.info(f"Analysis timeout: {settings.analysis_timeout_seconds}s")
    
    # Validate critical configuration
    if settings.llm_provider == "gemini" and not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY not configured!")
        raise ValueError("GEMINI_API_KEY must be set in environment variables")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Contract Risk Analysis System")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "llm_provider": settings.llm_provider,
        "max_file_size_mb": settings.max_file_size_mb
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Contract Risk Analysis System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Import and include API routes (will be implemented in later tasks)
# from api.routes import router as api_router
# app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.app_env == "development"),
        log_level=settings.log_level.lower()
    )
