@echo off
echo Starting AI Trading Bot (Crypto Sandbox - BTC/USD)...
echo ================================================
echo Strategy: Crypto Ensemble (5Min)
echo Config: config/crypto.yaml
echo Models: models_crypto/
echo ================================================

:: Set environment variable to tell main.py which config/models to use
set TRADING_BOT_CONFIG=config/crypto.yaml
set TRADING_BOT_MODELS=models_crypto

uv run python src/main.py

pause
