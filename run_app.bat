@echo off
setlocal ENABLEDELAYEDEXPANSION

title Explainable AI Copilot - Business Decision Support
cd /d "%~dp0"

echo ============================================================
echo  Explainable AI Copilot for Business Decision Support
echo ============================================================
echo.

REM Check that app.py exists in this folder
if not exist "app.py" (
    echo ERROR: app.py was not found in this folder.
    echo Please place run_app.bat inside the main project folder next to app.py.
    echo.
    pause
    exit /b 1
)

REM Check Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Please install Python 3.10 or 3.11, then run this file again.
    echo Download: https://www.python.org/downloads/
    echo IMPORTANT: Tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Create local virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip quietly
echo Checking Python packages...
python -m pip install --upgrade pip >nul

REM Install requirements only if streamlit is missing
python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    if exist "requirements.txt" (
        echo Installing required packages from requirements.txt...
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo ERROR: Package installation failed.
            echo Try running this command manually:
            echo python -m pip install -r requirements.txt
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: requirements.txt was not found.
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
echo A browser window should open automatically.
echo If it does not open, copy the local URL shown below.
echo.

python -m streamlit run app.py

echo.
echo App closed.
pause
