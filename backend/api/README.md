# API Routes Documentation

## Milestone 1 Endpoint

### POST /api/v1/analyze/milestone1

Analyzes a contract document using ML-based risk classification.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF or TXT)

**Response:**
```json
{
  "document_id": "uuid",
  "clauses": [
    {
      "id": "clause-uuid",
      "text": "The parties agree...",
      "risk_label": "high_risk",
      "confidence": 0.87,
      "position": 0
    }
  ],
  "summary": {
    "high_risk": 2,
    "medium_risk": 5,
    "low_risk": 10,
    "no_risk": 15
  }
}
```

**Error Responses:**

- 400 Bad Request: Invalid file type
- 413 Payload Too Large: File size exceeds limit
- 422 Unprocessable Entity: Parse error or segmentation error
- 500 Internal Server Error: Unexpected error
- 503 Service Unavailable: Model not trained

## Testing the Endpoint

### Using curl:

```bash
# Test with a PDF file
curl -X POST http://localhost:8000/api/v1/analyze/milestone1 \
  -F "file=@/path/to/contract.pdf"

# Test with a TXT file
curl -X POST http://localhost:8000/api/v1/analyze/milestone1 \
  -F "file=@/path/to/contract.txt"
```

### Using Python requests:

```python
import requests

url = "http://localhost:8000/api/v1/analyze/milestone1"

with open("contract.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(response.json())
```

### Using the FastAPI docs:

1. Start the server: `uvicorn main:app --reload`
2. Open browser: http://localhost:8000/docs
3. Find the `/api/v1/analyze/milestone1` endpoint
4. Click "Try it out"
5. Upload a file and execute

## Validation Rules

1. **File Type**: Only PDF and TXT files are accepted
2. **File Size**: Maximum 10MB (configurable via MAX_FILE_SIZE_MB env var)
3. **Content**: File must contain readable text (minimum 10 characters)

## Error Handling

The endpoint implements comprehensive error handling:

- File type validation before processing
- File size validation to prevent memory issues
- Descriptive error messages with suggestions
- Proper HTTP status codes
- Detailed error logging for debugging

## Requirements Implemented

- ✅ 1.1: Text file upload and extraction
- ✅ 1.2: PDF file upload and extraction
- ✅ 1.3: Descriptive error messages
- ✅ 9.1: File size limit enforcement
- ✅ 9.2: Retry logic (in classifier)
- ✅ 9.4: Error display without crashing
- ✅ 12.1: File upload widget support
- ✅ 12.2: Upload progress support
