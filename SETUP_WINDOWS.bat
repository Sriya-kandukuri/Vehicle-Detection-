@echo off
setlocal
title VehicleVision AI - Setup
echo =====================================================
echo   VehicleVision AI - First-time Windows setup
echo =====================================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.10 or 3.11 first.
  pause
  exit /b 1
)
if not exist venv (
  py -3.10 -m venv venv
  if errorlevel 1 py -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ai.txt
pip install -r requirements-dev.txt
if errorlevel 1 (
  echo Setup failed. Review the error above.
  pause
  exit /b 1
)
echo.
echo Setup completed successfully.
echo Accurate YOLOv8 detection is enabled by default.
echo Keep internet connected during the first prediction so yolov8n.pt can download.
echo Run RUN_WINDOWS.bat to start the application.
pause
