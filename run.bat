@echo off
REM SAM 2 Video Segmentation - Windows Startup Script
REM Run this file to start the application

echo.
echo ============================================
echo SAM 2 Video Segmentation
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies (if needed)
echo Installing/updating dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Start Streamlit app
echo.
echo ============================================
echo Starting SAM 2 Application...
echo ============================================
echo.
echo Opening http://localhost:8501 in your browser...
echo Press Ctrl+C to stop the application
echo.

streamlit run app/main.py --logger.level=info

pause
