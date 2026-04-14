# Contract Risk Analysis System - Backend

Backend API for contract risk analysis using ML and AI.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_api_key_here

# Run server
python3 main.py
```

Server runs at http://localhost:8000

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/analyze/milestone1` - ML-based risk classification
- `POST /api/v1/analyze/milestone2` - Agentic risk analysis

## Configuration

Edit `.env` file for configuration. Key variable:
- `GEMINI_API_KEY` - Required for AI analysis
