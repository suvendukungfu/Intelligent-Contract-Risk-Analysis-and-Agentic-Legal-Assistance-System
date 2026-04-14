@echo off
REM Setup script for Contract Risk Analysis System backend (Windows)

echo Setting up Contract Risk Analysis System Backend...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3 is not installed. Please install Python 3.11 or higher.
    exit /b 1
)

echo Python found
python --version

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env and add your GEMINI_API_KEY
    echo Get your API key from: https://makersuite.google.com/app/apikey
    echo.
) else (
    echo .env file already exists
)

REM Create necessary directories
echo Creating necessary directories...
if not exist "ml\models" mkdir ml\models
if not exist "data\vector_store" mkdir data\vector_store
if not exist "data\legal_documents" mkdir data\legal_documents
if not exist "data\training_data" mkdir data\training_data
if not exist "logs" mkdir logs

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env and add your GEMINI_API_KEY
echo 2. Activate the virtual environment: venv\Scripts\activate.bat
echo 3. Run the application: python main.py
echo 4. Visit http://localhost:8000/docs for API documentation
echo.

pause
