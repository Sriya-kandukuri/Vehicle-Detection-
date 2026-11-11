@echo off
setlocal
title VehicleVision AI
if not exist venv\Scripts\python.exe (
  echo Virtual environment not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
set FLASK_APP=app.py
echo Starting VehicleVision AI at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause

