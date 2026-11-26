#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then echo "Python 3.10+ is required."; exit 1; fi
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate
if ! python -c "import customtkinter, cv2, scipy, imageio_ffmpeg, tkinterdnd2" >/dev/null 2>&1; then
  echo "Installing requirements (first run only)..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi
python main.py "$@"
