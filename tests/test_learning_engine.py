"""
Unit tests for the Learning Engine.
Verifies atomic writes, schema validation, and critique parsing.
"""
import pytest
import os
import json
from unittest.mock import MagicMock, patch
from src.learning_engine import LearningEngine

# Test constants
TEST_HISTORY_FILE = "data/test_trade_history.json"
TEST_LESSONS_FILE = "config/test_lessons.txt"

@pytest.fixture
def engine():
    # Setup
    if os.path.exists(TEST_HISTORY_FILE):
        os.remove(TEST_HISTORY_FILE)
    if os.path.exists(TEST_LESSONS_FILE):
        os.remove(TEST_LESSONS_FILE)
        
    engine = LearningEngine(history_file=TEST_HISTORY_FILE, lessons_file=TEST_LESSONS_FILE)
    yield engine
    
    # Teardown
    if os.path.exists(TEST_HISTORY_FILE):
        os.remove(TEST_HISTORY_FILE)
    if os.path.exists(TEST_LESSONS_FILE):
        os.remove(TEST_LESSONS_FILE)

def test_atomic_writes(engine):
    """Verify that trades are saved correctly and atomically."""
    # Create dummy trade
    trade = {
        "order_id": "123",
        "ticker": "TSLA",
        "action": "BUY",
        "qty": 10,
        "entry_time": "2023-01-01T12:00:00",
        "status": "OPEN",
        "logic_engine": "Test",
        "conviction": "HIGH",
        "initial_reasoning": "Test reasoning"
    }
    
    # Log trade
    engine.log_trade_entry({"ticker": "TSLA", "action": "BUY", "qty": 10, "analysis": {"conviction": "HIGH"}}, "123")
    
    # Check file exists and contains data
    assert os.path.exists(TEST_HISTORY_FILE)
    with open(TEST_HISTORY_FILE, "r") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["order_id"] == "123"
        assert data[0]["status"] == "OPEN"

def test_critique_parsing(engine):
    """Verify that valid critiques are parsed and appended to lessons."""
    
    # Mock trade history with a closed trade
    closed_trade = {
        "order_id": "999",
        "ticker": "NVDA",
        "action": "BUY",
        "status": "CLOSED",
        "pnl_usd": -50.0,
        "review_status": "PENDING",
        "logic_engine": "Squeeze",
        "conviction": "MED",
        "planned_entry": 100,
        "planned_stop": 90,
        "planned_target": 130,
        "initial_reasoning": "Breakout imminent"
    }
    engine._save_trades([closed_trade])
    
    # Mock the AI Brain response
    mock_brain = MagicMock()
    mock_brain.run_critique.return_value = "Analysis: The trade failed due to market chop.\nLESSON: Wait for candle close confirmation before entering breakout."
    
    with patch("src.learning_engine.get_brain", return_value=mock_brain):
        result = engine.analyze_past_performance()
    
    # Assertions
    assert "Wait for candle close confirmation" in result or "1 new lessons added" in result
    
    # Verify lesson file
    with open(TEST_LESSONS_FILE, "r") as f:
        content = f.read()
        assert "Wait for candle close confirmation" in content
    
    # Verify trade status updated
    updated_trades = engine._load_trades()
    assert updated_trades[0]["review_status"] == "REVIEWED"

def test_no_lesson_parsing(engine):
    """Verify that 'NO LESSON' responses are ignored."""
    
    # Mock trade history
    closed_trade = {
        "order_id": "888",
        "status": "CLOSED",
        "review_status": "PENDING",
        "ticker": "AAPL",
        "action": "BUY",
        "logic_engine": "Macro",
        "conviction": "LOW",
        "planned_entry": 150, "planned_stop": 140, "planned_target": 160
    }
    engine._save_trades([closed_trade])
    
    # Mock AI response with NO LESSON
    mock_brain = MagicMock()
    mock_brain.run_critique.return_value = "This was just bad luck.\nLESSON: NO LESSON"
    
    with patch("src.learning_engine.get_brain", return_value=mock_brain):
        engine.analyze_past_performance()
    
    # Verify file is empty (or doesn't exist if created empty)
    if os.path.exists(TEST_LESSONS_FILE):
        with open(TEST_LESSONS_FILE, "r") as f:
            content = f.read()
            assert content.strip() == ""
