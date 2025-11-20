@echo off
cd /d "%~dp0"
echo ========================================
echo   TV Segment & Price Advisor
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo [1/2] Installing/updating dependencies...
cd backend
pip install -r requirements.txt --quiet
cd ..

echo.
echo [2/2] Starting FastAPI server...
echo.
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python main.py

pause

