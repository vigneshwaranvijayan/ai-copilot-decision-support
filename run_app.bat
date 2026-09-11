@echo off
setlocal EnableExtensions

title Explainable AI Copilot - Fast Launcher
cd /d "%~dp0"

if not exist "app.py" (
    echo ERROR: app.py was not found next to run_app.bat.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.10 or 3.11 and tick "Add Python to PATH".
    pause
    exit /b 1
)

REM Reuse one environment across all extracted versions of this project.
REM This prevents a fresh multi-minute reinstall whenever a corrected ZIP is extracted.
if "%LOCALAPPDATA%"=="" (
    set "COPILOT_RUNTIME=%USERPROFILE%\.dataset_grounded_ai_copilot"
) else (
    set "COPILOT_RUNTIME=%LOCALAPPDATA%\DatasetGroundedAICopilot"
)
set "VENV_DIR=%COPILOT_RUNTIME%\venv"
set "READY_FILE=%COPILOT_RUNTIME%\.runtime_ready"

if not exist "%COPILOT_RUNTIME%" mkdir "%COPILOT_RUNTIME%" >nul 2>nul

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo First-time setup only: creating shared Python environment...
    python -m venv --system-site-packages "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create the shared environment.
        pause
        exit /b 1
    )
)

if not exist "%READY_FILE%" (
    echo First-time setup only: checking required packages...
    "%VENV_DIR%\Scripts\python.exe" -c "import streamlit,pandas,numpy,sklearn,plotly,openpyxl,requests,rapidfuzz,joblib,tabulate,psutil" >nul 2>nul
    if errorlevel 1 (
        echo Installing missing core packages once. Future launches and future extracted versions will reuse them.
        "%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --prefer-binary -r requirements.txt
        if errorlevel 1 (
            echo ERROR: Core package installation failed.
            pause
            exit /b 1
        )
    )

    REM SHAP is needed for the assessed explanation workflow. Install only if missing.
    "%VENV_DIR%\Scripts\python.exe" -c "import shap" >nul 2>nul
    if errorlevel 1 (
        echo Installing SHAP once for explanation support...
        "%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --prefer-binary "shap>=0.44"
        if errorlevel 1 (
            echo WARNING: SHAP installation failed. The app can still use fallback feature importance.
        )
    )

    > "%READY_FILE%" echo ready
)

echo Starting AI Copilot...
"%VENV_DIR%\Scripts\python.exe" -m streamlit run app.py --server.fileWatcherType none --browser.gatherUsageStats false

endlocal
