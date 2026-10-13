@echo off
rem Creates .venv and installs Chibi Companion into it (run once; safe to re-run).
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Python 3.12+ is required: https://www.python.org/downloads/
        exit /b 1
    )
)
.venv\Scripts\python.exe -m pip install --upgrade pip
rem The voice extra (Whisper, onnxruntime, sounddevice) is ~300 MB; models are downloaded later
rem from Settings - Voice.  Without it everything except voice works.
.venv\Scripts\python.exe -m pip install -e ".[voice]"
if errorlevel 1 (
    echo Installing dependencies failed.
    exit /b 1
)
echo Done. Start the app with run.bat
