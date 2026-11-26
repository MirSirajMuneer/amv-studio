@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/ and tick "Add to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -c "import customtkinter, cv2, scipy, imageio_ffmpeg, tkinterdnd2" >nul 2>nul
if errorlevel 1 (
  echo Installing requirements, this only happens once...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
)
python main.py
if errorlevel 1 pause
