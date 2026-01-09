@echo off
REM Switch to the directory where this script is located
cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

echo =================================================== > logs\daily_run_LAST.log
echo Run started at %date% %time% >> logs\daily_run_LAST.log
echo =================================================== >> logs\daily_run_LAST.log

echo Running Auto Trading Bot...
echo Logging output to logs\daily_run_LAST.log

REM Execute the bot and redirect stdout and stderr to the log file
uv run python src/live_trading.py >> logs\daily_run_LAST.log 2>&1

REM Check the exit code
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The bot crashed or exited with an error code: %ERRORLEVEL%
    echo Check logs\daily_run_LAST.log for more details.
    echo.
    echo Press any key to close this window...
    pause
) else (
    echo.
    echo Bot finished successfully (or was stopped manually).
)

