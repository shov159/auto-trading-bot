#!/bin/bash
echo "Starting AI Trading Bot (RL Ensemble Mode)..."
echo "================================================"
echo "Risk Manager: ACTIVE"
echo "Strategy: PPO (Anchor) + A2C (Reactor) + DDPG (Sniper)"
echo "================================================"

uv run python src/main.py

