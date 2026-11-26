@echo off
REM Builds a standalone Windows app (dist\AMV_Studio\AMV_Studio.exe) with PyInstaller.
REM NOTE: this script has NOT been tested by the author - try it on a clean machine before distributing.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name AMV_Studio ^
  --collect-all customtkinter --collect-all imageio_ffmpeg --collect-all tkinterdnd2 ^
  --hidden-import amv_studio.pipeline --hidden-import amv_studio.worker ^
  main.py
echo.
echo Done. Test dist\AMV_Studio\AMV_Studio.exe, then zip the dist\AMV_Studio folder (or wrap it with Inno Setup).
pause
