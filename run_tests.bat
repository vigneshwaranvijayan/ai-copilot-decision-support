@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_app.bat once first so the local environment exists.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import pytest" >nul 2>nul
if errorlevel 1 (
  echo Installing test-only dependency pytest...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check pytest>=7.4
)
".venv\Scripts\python.exe" -m pytest -q
pause
