@echo off
rem Starts Chibi Companion without a console window.
rem Logs: %APPDATA%\ChibiCompanion\logs\chibi.log
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    call setup.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
start "" .venv\Scripts\pythonw.exe -m chibi %*
