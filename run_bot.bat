@echo off
REM Switch to the directory where this script is located
cd /d "%~dp0"
if not exist logs mkdir logs
echo =================================================== > logs\daily_run_LAST.log
echo Run started at %date% %time% >> logs\daily_run_LAST.log
uv run python src/main.py >> logs\daily_run_LAST.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The bot crashed or exited with an error code: %ERRORLEVEL%
    pause
)
