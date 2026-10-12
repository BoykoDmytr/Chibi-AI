@echo off
rem Starts Chibi Companion with a console and debug logging.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    call setup.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
.venv\Scripts\python.exe -m chibi --debug %*
pause
