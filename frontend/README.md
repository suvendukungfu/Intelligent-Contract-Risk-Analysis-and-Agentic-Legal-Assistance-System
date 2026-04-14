# Contract Risk Analysis System - Frontend

React + TypeScript frontend for the Contract Risk Analysis System.

## Features

- File upload with drag-and-drop support (PDF and TXT files)
- Milestone 1: ML-based risk classification with visual highlighting
- Milestone 2: Agentic AI risk analysis with comprehensive reports
- Responsive design with custom CSS
- Real-time progress indicators

## Tech Stack

- React 18
- TypeScript
- Vite
- Axios (API client)
- React Dropzone (file upload)

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
npm install
```

### Configuration

Copy `.env.example` to `.env` and configure the API base URL:

```bash
cp .env.example .env
```

Edit `.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

### Development

Start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Build

Build for production:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # API client and type definitions
│   ├── components/
│   │   ├── Header.tsx          # Application header
│   │   ├── MilestoneSelector.tsx  # Milestone navigation
│   │   ├── FileUpload.tsx      # File upload component
│   │   ├── RiskVisualization.tsx  # Milestone 1 results
│   │   ├── ReportView.tsx      # Milestone 2 results
│   │   ├── Loading.tsx         # Loading indicator
│   │   └── ErrorMessage.tsx    # Error display
│   ├── App.tsx                 # Main application component
│   ├── main.tsx                # Application entry point
│   └── index.css               # Global styles
├── .env                        # Environment variables
└── package.json
```

## API Integration

The frontend communicates with the backend API through two main endpoints:

- `POST /api/v1/analyze/milestone1` - ML-based classification
- `POST /api/v1/analyze/milestone2` - Agentic AI analysis

See `src/api/client.ts` for API client implementation and type definitions.

## Components

### FileUpload
Drag-and-drop file upload component supporting PDF and TXT files.

### MilestoneSelector
Toggle between Milestone 1 (ML Classification) and Milestone 2 (Agentic Analysis).

### RiskVisualization
Displays contract text with color-coded risk highlighting and filtering options.

### ReportView
Displays comprehensive risk analysis report with severity badges and mitigation actions.

## Styling

The application uses custom CSS (no framework) for styling. All styles are defined in `src/index.css`.

Color scheme:
- High Risk: Red (#ef5350)
- Medium Risk: Orange (#ff9800)
- Low Risk: Yellow (#fdd835)
- No Risk: Green (#66bb6a)
