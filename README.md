# Contract Risk Analysis System

AI-powered legal document analysis with intelligent risk detection.

## Features

- **ML Classification**: Automated risk classification using machine learning
- **Agentic Analysis**: Intelligent AI agent for comprehensive risk assessment
- **RAG System**: Legal guideline retrieval for context-aware analysis
- **Dark Theme UI**: Professional, industry-standard interface

## Tech Stack

**Backend:**
- FastAPI
- Python 3.11+
- Gemini AI
- ChromaDB
- scikit-learn

**Frontend:**
- React + TypeScript
- Vite
- Axios

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

Run server:
```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Open http://localhost:5173
2. Select analysis mode (ML Classification or Agentic Analysis)
3. Upload contract (PDF or TXT)
4. View risk analysis results

## License

MIT
