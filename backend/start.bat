@echo off
echo.
echo  OSB - Operating System Brain
echo  ==============================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

REM Copy .env if not exists
if not exist ".env" (
    copy .env.example .env
    echo.
    echo  ACTION REQUIRED:
    echo  Open .env file and add your GEMINI_API_KEY
    echo  Get free key at: https://makersuite.google.com/app/apikey
    echo.
    pause
)

REM Start server
echo.
echo  Starting OSB backend on http://localhost:8000
echo  API docs: http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo.
python main.py
