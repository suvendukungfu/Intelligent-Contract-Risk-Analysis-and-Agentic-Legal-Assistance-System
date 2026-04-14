# Contract Risk Analysis System - Backend

Backend API for the Intelligent Contract Risk Analysis and Agentic Legal Assistance System.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` and set the required variables:

**Required:**
- `GEMINI_API_KEY`: Your Google Gemini API key (get it from https://makersuite.google.com/app/apikey)

**Optional (defaults provided):**
- `MAX_FILE_SIZE_MB`: Maximum upload file size (default: 10)
- `ANALYSIS_TIMEOUT_SECONDS`: Timeout for agentic analysis (default: 60)
- `LLM_MODEL`: Gemini model to use (default: gemini-1.5-flash)

### 4. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Environment Variables

### Application Configuration

- `APP_ENV`: Environment (development/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `MAX_FILE_SIZE_MB`: Maximum file upload size in MB
- `ANALYSIS_TIMEOUT_SECONDS`: Timeout for analysis operations

### Server Configuration

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins

### ML Models Configuration

- `MODEL_PATH`: Path to ML model files
- `CLASSIFIER_MODEL`: Risk classifier model filename
- `EMBEDDING_MODEL`: Sentence transformer model name

### LLM Configuration

- `LLM_PROVIDER`: LLM provider (gemini)
- `GEMINI_API_KEY`: Google Gemini API key (required)
- `LLM_MODEL`: Model name (gemini-1.5-flash or gemini-1.5-pro)
- `LLM_MAX_TOKENS`: Maximum tokens in LLM response
- `LLM_TEMPERATURE`: Temperature for LLM generation

### RAG Configuration

- `VECTOR_STORE_PATH`: Path to vector store data
- `VECTOR_STORE_TYPE`: Vector store type (chromadb)
- `RAG_TOP_K`: Number of documents to retrieve
- `CHUNK_SIZE`: Document chunk size for RAG
- `CHUNK_OVERLAP`: Overlap between chunks

### File Upload Configuration

- `ALLOWED_EXTENSIONS`: Comma-separated allowed file extensions
- `MAX_UPLOAD_SIZE_BYTES`: Maximum upload size in bytes

### Retry Configuration

- `MAX_RETRIES`: Maximum retry attempts for LLM calls
- `RETRY_DELAY_SECONDS`: Delay between retries

## API Endpoints

### Health Check

```bash
GET /health
```

### Milestone 1: ML-Based Risk Classification

```bash
POST /api/v1/analyze/milestone1
Content-Type: multipart/form-data

file: <contract.pdf or contract.txt>
```

### Milestone 2: Agentic Risk Analysis

```bash
POST /api/v1/analyze/milestone2
Content-Type: multipart/form-data

file: <contract.pdf or contract.txt>
```

## Development

### Run Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Format Code

```bash
black .
```

### Lint Code

```bash
flake8 .
```

## Project Structure

```
backend/
├── api/              # API routes and models
├── core/             # Core business logic
├── ml/               # ML training and models
├── data/             # Data storage
├── tests/            # Test files
└── main.py           # Application entry point
```
