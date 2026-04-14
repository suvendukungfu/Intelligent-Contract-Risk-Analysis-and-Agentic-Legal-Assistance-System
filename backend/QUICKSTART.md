# Quick Start Guide

Get the Contract Risk Analysis System backend up and running in minutes.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Google Gemini API key (free tier available)

## Getting Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## Setup (Automated)

### Linux/macOS

```bash
cd backend
./scripts/setup.sh
```

### Windows

```cmd
cd backend
scripts\setup.bat
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies
- Download required NLP models
- Create a `.env` file from the template
- Create necessary directories

## Setup (Manual)

If you prefer manual setup:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Create .env file
cp .env.example .env

# 6. Create directories
mkdir -p ml/models data/vector_store data/legal_documents data/training_data logs
```

## Configuration

Edit the `.env` file and add your Gemini API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

Optional: Adjust other settings as needed:

```bash
# Increase file size limit to 20MB
MAX_FILE_SIZE_MB=20

# Increase timeout to 90 seconds
ANALYSIS_TIMEOUT_SECONDS=90

# Use Gemini Pro instead of Flash
LLM_MODEL=gemini-1.5-pro
```

## Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the server
python main.py
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Testing the Setup

### 1. Check Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "environment": "development",
  "llm_provider": "gemini",
  "max_file_size_mb": 10
}
```

### 2. Run Tests

```bash
pytest
```

### 3. Check Coverage

```bash
pytest --cov=. --cov-report=html
```

Open `htmlcov/index.html` in your browser to view the coverage report.

## Common Issues

### Issue: "GEMINI_API_KEY must be set"

**Solution:** Make sure you've added your API key to the `.env` file:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### Issue: "ModuleNotFoundError: No module named 'pydantic_settings'"

**Solution:** Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Can't find model 'en_core_web_sm'"

**Solution:** Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Port 8000 already in use

**Solution:** Change the port in `.env`:
```bash
PORT=8001
```

## Next Steps

1. Implement document parser (Task 6)
2. Implement clause segmenter (Task 7)
3. Prepare training data (Task 8)
4. Train risk classifier (Task 9)
5. Create API endpoints (Task 10)

## Development Workflow

```bash
# Activate virtual environment
source venv/bin/activate

# Run with auto-reload (development)
uvicorn main:app --reload

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Type check
mypy .
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

## Support

For issues or questions:
1. Check the main README.md
2. Review the design document at `.kiro/specs/contract-risk-analysis-system/design.md`
3. Check the requirements at `.kiro/specs/contract-risk-analysis-system/requirements.md`
