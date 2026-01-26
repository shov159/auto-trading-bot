@echo off
echo Starting CIO Trading Bot...
echo.
cd /d "%~dp0"
python -m src.telegram_bot
pause

