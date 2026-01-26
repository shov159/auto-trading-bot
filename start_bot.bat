@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: CIO Trading Bot - One-Click Windows Launcher
:: =============================================================================
:: Double-click this file to:
:: 1. Create/activate virtual environment
:: 2. Install dependencies
:: 3. Validate API keys
:: 4. Start the bot
:: =============================================================================

title CIO Trading Bot Launcher

:: Colors for Windows 10+
set "GREEN=[92m"
set "RED=[91m"
set "CYAN=[96m"
set "YELLOW=[93m"
set "RESET=[0m"

echo.
echo %CYAN%======================================================================%RESET%
echo %CYAN%            CIO TRADING BOT - ONE-CLICK LAUNCHER                    %RESET%
echo %CYAN%======================================================================%RESET%
echo.

:: Navigate to script directory
cd /d "%~dp0"
echo %CYAN%[1/5]%RESET% Working directory: %CD%

:: =============================================================================
:: CHECK PYTHON
:: =============================================================================
echo.
echo %CYAN%[2/5]%RESET% Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo Please install Python 3.11+ from https://python.org
    goto :error
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Found Python %PYTHON_VERSION%

:: =============================================================================
:: VIRTUAL ENVIRONMENT
:: =============================================================================
echo.
echo %CYAN%[3/5]%RESET% Setting up virtual environment...

if not exist "venv" (
    echo       Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo %RED%ERROR: Failed to create virtual environment%RESET%
        goto :error
    )
    echo       %GREEN%Virtual environment created%RESET%
    set FRESH_VENV=1
) else (
    echo       %GREEN%Virtual environment found%RESET%
    set FRESH_VENV=0
)

:: Activate virtual environment
echo       Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%ERROR: Failed to activate virtual environment%RESET%
    goto :error
)
echo       %GREEN%Virtual environment activated%RESET%

:: =============================================================================
:: INSTALL DEPENDENCIES
:: =============================================================================
echo.
echo %CYAN%[4/5]%RESET% Installing dependencies...

if "%FRESH_VENV%"=="1" (
    echo       Upgrading pip...
    python -m pip install --upgrade pip -q
)

:: Check if requirements need to be installed
pip show colorama >nul 2>&1
if errorlevel 1 (
    echo       Installing packages from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo %RED%ERROR: Failed to install dependencies%RESET%
        goto :error
    )
    echo       %GREEN%Dependencies installed%RESET%
) else (
    echo       %GREEN%Dependencies already installed%RESET%
)

:: =============================================================================
:: VALIDATE API KEYS
:: =============================================================================
echo.
echo %CYAN%[5/5]%RESET% Validating API keys...
echo.

python validate_keys.py
if errorlevel 1 (
    echo.
    echo %RED%======================================================================%RESET%
    echo %RED%        API KEY VALIDATION FAILED - Please fix errors above          %RESET%
    echo %RED%======================================================================%RESET%
    echo.
    echo Make sure your .env file contains:
    echo   TELEGRAM_TOKEN=your_bot_token
    echo   TELEGRAM_CHAT_ID=your_chat_id
    echo   OPENAI_API_KEY=sk-your-key
    echo   ALPACA_API_KEY=your_alpaca_key
    echo   ALPACA_SECRET_KEY=your_alpaca_secret
    echo.
    goto :error
)

:: =============================================================================
:: START THE BOT
:: =============================================================================
echo.
echo %GREEN%======================================================================%RESET%
echo %GREEN%                    STARTING CIO TRADING BOT                         %RESET%
echo %GREEN%======================================================================%RESET%
echo.
echo Press Ctrl+C to stop the bot
echo.

python run.py

:: If we get here, the bot exited
echo.
echo %YELLOW%Bot has stopped.%RESET%
goto :end

:: =============================================================================
:: ERROR HANDLER
:: =============================================================================
:error
echo.
echo %RED%Setup failed. Please check the errors above.%RESET%
echo.

:end
echo.
echo Press any key to close this window...
pause >nul

