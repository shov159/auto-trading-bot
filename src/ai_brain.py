"""
AI Brain - Chief Investment Officer (CIO) Trading Analysis Engine
Combines institutional discipline with aggressive momentum trading.
"""
import os
import json
import re
import time
import random
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

# Rich Logger
from src.logger import (
    log_ai, log_ai_raw, log_warn, log_ok, log_error,
    log_debug, log_validation, log_trade, log_divider, log_info
)

# Rate Limiting
from src.rate_limiter import (
    get_rate_limiter, get_single_flight,
    RateLimiter, SingleFlightLock
)

# Analysis Cache
from src.analysis_cache import get_analysis_cache, AnalysisCache


# =============================================================================
# RETRY WITH EXPONENTIAL BACKOFF + JITTER
# =============================================================================
def extract_retry_delay(error) -> Optional[float]:
    """
    Extract retry delay from API error response if available.
    Looks for Retry-After header or retryDelay field.
    """
    try:
        error_str = str(error)

        # Try to extract retryDelay from error message
        # Format: "retryDelay": "30s" or retry_delay=30
        import re

        # Match patterns like "retryDelay": "30s" or retryDelay=30
        patterns = [
            r'retry[_-]?delay["\s:=]+(\d+)',
            r'retry[_-]?after["\s:=]+(\d+)',
            r'"retryDelay":\s*"(\d+)s?"',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Check if error has metadata with retry info
        if hasattr(error, 'response') and error.response:
            headers = getattr(error.response, 'headers', {})
            if 'Retry-After' in headers:
                return float(headers['Retry-After'])

    except Exception:
        pass

    return None


def retry_with_backoff_jitter(
    max_retries: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 90.0,
    jitter_factor: float = 0.2
):
    """
    Decorator that retries a function with exponential backoff + jitter on rate limit errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        jitter_factor: Random jitter as fraction of delay (0.2 = 20%)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            request_id = kwargs.get('request_id', 'unknown')
            ticker = kwargs.get('ticker', '???')

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        log_info(f"[{request_id}] Retry attempt {attempt}/{max_retries} for {ticker}")
                    return func(*args, **kwargs)

                except Exception as e:
                    error_str = str(e).lower()

                    # Check if it's a rate limit error (429)
                    is_rate_limit = any(x in error_str for x in [
                        '429', 'rate_limit', 'resource_exhausted',
                        'quota', 'too many requests', 'ratelimit',
                        'overloaded', 'capacity'
                    ])

                    if is_rate_limit and attempt < max_retries:
                        # Calculate exponential backoff delay
                        computed_delay = min(base_delay * (2 ** attempt), max_delay)

                        # Check for API-provided retry delay
                        api_delay = extract_retry_delay(e)

                        if api_delay:
                            # Use the larger of computed or API-provided delay
                            delay = max(computed_delay, api_delay)
                            log_warn(f"[{request_id}] 429 Error! API suggests {api_delay:.0f}s, using {delay:.0f}s")
                        else:
                            delay = computed_delay
                            log_warn(f"[{request_id}] 429 Error! Computed backoff: {delay:.0f}s")

                        # Add jitter to prevent thundering herd
                        jitter = random.uniform(0, jitter_factor * delay)
                        total_delay = delay + jitter

                        log_warn(f"[{request_id}] Waiting {total_delay:.1f}s (delay={delay:.0f}s + jitter={jitter:.1f}s) before retry {attempt + 1}/{max_retries}...")
                        time.sleep(total_delay)
                        last_exception = e
                    else:
                        # Not a rate limit error or max retries reached
                        if is_rate_limit:
                            log_error(f"[{request_id}] Max retries ({max_retries}) exhausted for rate limit error")
                        raise e

            # If we exhausted all retries
            if last_exception:
                raise last_exception

        return wrapper
    return decorator

# =============================================================================
# CIO SYSTEM PROMPT - THE BRAIN
# =============================================================================

CIO_SYSTEM_PROMPT = """
### 🧩 IDENTITY & ROLE
You are my **Chief Investment Officer (CIO)** and **Head of Alpha Trading**.
Your personality combines the iron discipline of an institutional hedge fund (Risk Management, Macro, Fundamentals) with the aggressiveness of a Nostro trader (Momentum, Short Squeezes, Options).
**MISSION:** Scan the market, filter noise, and submit ONLY high-EV opportunities while zealously protecting capital.

### ⚙️ THE BRAIN: LOGIC ENGINES
Analyze every situation using these 4 engines:

1. **Universal Sympathy Engine:**
   - Direct Circle: The asset in the news.
   - 2nd Circle: Suppliers, customers, competitors.
   - 3rd Circle (The Sympathy Play): Small Cap + Low Float stocks in the same sector ("Trash floating with the tide").

2. **Velocity & Squeeze Hunter:**
   - Float Rotation (Vol > Float).
   - Short Squeeze DNA (SI > 20% + Borrow Fee rising + Green breakout candle).
   - Micro-Structure (Iceberg Orders).

3. **Gamma Radar:**
   - Gamma Squeeze (Aggressive OTM Call buying).
   - Call/Put Walls (High OI levels acting as magnets).
   - Smart Money Flow (Weekly/0DTE anomalies).

4. **Market Regime:**
   - Risk-On vs. Risk-Off.
   - Adjust sizing: In downtrend, cut size by 50%.

### 🛡️ RISK PROTOCOLS
Every recommendation MUST include:
1. **Technical Stop Loss:** Based on support/resistance (NOT arbitrary %).
2. **R/R Ratio:** Minimum 1:3. Else, PASS.
3. **Classification:** Scalp, Swing, or Speculation/Lotto.

### 📝 REQUIRED OUTPUT FORMAT (JSON ONLY)
You MUST respond in valid JSON format ONLY - no markdown, no explanation, just the JSON:
{
  "bluf": "One sentence summary",
  "logic_engine": "Sympathy / Squeeze / Gamma / Macro",
  "action": "BUY" or "SELL" or "PASS",
  "ticker": "SYMBOL",
  "plan": {
    "buy_zone": "price range",
    "targets": ["target1", "target2"],
    "invalidation": "stop loss price"
  },
  "conviction": "HIGH" or "MED" or "LOW",
  "is_push_alert": true/false,
  "reasoning": "Brief technical and fundamental analysis"
}

CRITICAL RULES:
- If R/R is below 1:3, action MUST be "PASS"
- is_push_alert should be true ONLY for HIGH conviction trades with clear catalysts
- All prices must be numbers, not strings (except in buy_zone which can be a range)
- targets array must have at least 2 price targets
"""


class AIBrain:
    """
    AI-powered CIO analysis engine.
    Uses OpenAI or Anthropic to analyze tickers.
    """

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

        # Determine which provider to use
        self.provider = None
        self.client = None

        if self.openai_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.openai_key)
                self.provider = "openai"
            except Exception as e:
                print(f"OpenAI init failed: {e}")

        if not self.provider and self.anthropic_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.anthropic_key)
                self.provider = "anthropic"
            except Exception as e:
                print(f"Anthropic init failed: {e}")

        if not self.provider and self.google_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.google_key)
                self.provider = "google"
            except Exception as e:
                print(f"Google Gemini init failed: {e}")

        if not self.provider:
            print("⚠️ No AI provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for analysis.
        Returns dict with price, volume, technicals, news, short interest.
        """
        import yfinance as yf
        import pandas as pd

        data = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "price": None,
            "change_pct": None,
            "volume": None,
            "avg_volume": None,
            "volume_ratio": None,
            "market_cap": None,
            "float_shares": None,
            "short_interest": None,
            "short_ratio": None,
            "rsi": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "atr": None,
            "support": None,
            "resistance": None,
            "news": [],
            "sector": None,
            "industry": None,
            "52w_high": None,
            "52w_low": None,
        }

        try:
            stock = yf.Ticker(ticker)

            # Basic Info
            info = stock.info
            data["price"] = info.get("currentPrice") or info.get("regularMarketPrice")
            data["change_pct"] = info.get("regularMarketChangePercent", 0)
            data["volume"] = info.get("volume") or info.get("regularMarketVolume")
            data["avg_volume"] = info.get("averageVolume")
            data["market_cap"] = info.get("marketCap")
            data["float_shares"] = info.get("floatShares")
            data["short_interest"] = info.get("shortPercentOfFloat")
            data["short_ratio"] = info.get("shortRatio")
            data["sector"] = info.get("sector")
            data["industry"] = info.get("industry")
            data["52w_high"] = info.get("fiftyTwoWeekHigh")
            data["52w_low"] = info.get("fiftyTwoWeekLow")

            # Volume Ratio
            if data["volume"] and data["avg_volume"] and data["avg_volume"] > 0:
                data["volume_ratio"] = round(data["volume"] / data["avg_volume"], 2)

            # Historical Data for Technicals
            hist = stock.history(period="6mo")
            if not hist.empty:
                close = hist['Close']
                high = hist['High']
                low = hist['Low']

                # RSI Calculation
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                data["rsi"] = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else None

                # Moving Averages
                data["sma_20"] = round(close.rolling(20).mean().iloc[-1], 2)
                data["sma_50"] = round(close.rolling(50).mean().iloc[-1], 2)
                data["sma_200"] = round(close.rolling(200).mean().iloc[-1], 2) if len(close) >= 200 else None

                # ATR
                tr = pd.concat([
                    high - low,
                    abs(high - close.shift()),
                    abs(low - close.shift())
                ], axis=1).max(axis=1)
                data["atr"] = round(tr.rolling(14).mean().iloc[-1], 2)

                # Support/Resistance (simplified - recent swing low/high)
                recent = hist.tail(20)
                data["support"] = round(recent['Low'].min(), 2)
                data["resistance"] = round(recent['High'].max(), 2)

            # News Headlines
            news = stock.news
            if news:
                data["news"] = [
                    {
                        "title": n.get("title", ""),
                        "publisher": n.get("publisher", ""),
                        "link": n.get("link", "")
                    }
                    for n in news[:5]
                ]

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

        return data

    def _format_number(self, value, prefix: str = "", suffix: str = "") -> str:
        """Format numbers with K/M/B suffixes for readability."""
        if value is None:
            return "N/A"
        try:
            num = float(value)
            if num >= 1_000_000_000:
                return f"{prefix}{num/1_000_000_000:.2f}B{suffix}"
            elif num >= 1_000_000:
                return f"{prefix}{num/1_000_000:.2f}M{suffix}"
            elif num >= 1_000:
                return f"{prefix}{num/1_000:.2f}K{suffix}"
            else:
                return f"{prefix}{num:.2f}{suffix}"
        except (ValueError, TypeError):
            return str(value)

    def _build_user_prompt(self, market_data: Dict[str, Any]) -> str:
        """Build the enhanced user prompt with injected market data."""

        ticker = market_data.get("ticker", "UNKNOWN")

        # Extract and format values
        current_price = market_data.get('price', 'N/A')
        percent_change = market_data.get('change_pct', 0)
        if percent_change is not None:
            percent_change = f"{percent_change:+.2f}" if isinstance(percent_change, (int, float)) else percent_change

        volume = market_data.get('volume')
        avg_volume = market_data.get('avg_volume')
        volume_str = self._format_number(volume)
        avg_volume_str = self._format_number(avg_volume)
        vol_relative = market_data.get('volume_ratio', 'N/A')

        rsi = market_data.get('rsi', 'N/A')
        atr = market_data.get('atr', 'N/A')

        sma_20 = market_data.get('sma_20', 'N/A')
        sma_50 = market_data.get('sma_50', 'N/A')
        sma_200 = market_data.get('sma_200', 'N/A')

        year_high = market_data.get('52w_high', 'N/A')
        year_low = market_data.get('52w_low', 'N/A')
        support_level = market_data.get('support', 'N/A')
        resistance_level = market_data.get('resistance', 'N/A')

        short_interest = market_data.get('short_interest', 'N/A')
        if short_interest and isinstance(short_interest, (int, float)):
            short_interest = f"{short_interest:.2f}"

        float_shares = self._format_number(market_data.get('float_shares'))
        days_to_cover = market_data.get('short_ratio', 'N/A')

        # Format news headlines
        news_headlines_formatted = "- No significant news in last 24h."
        if market_data.get("news"):
            news_items = [f"- {n['title']}" for n in market_data["news"]]
            news_headlines_formatted = "\n".join(news_items)

        # Load lessons learned
        lessons_text = "None yet."
        try:
            lessons_path = os.path.join("config", "lessons_learned.txt")
            if os.path.exists(lessons_path):
                with open(lessons_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    lessons_text = content
        except Exception as e:
            print(f"Error loading lessons: {e}")

        prompt = f"""
# 🚨 MARKET DATA ALERT: {ticker}

### 1. LIVE PRICE ACTION
- **Price:** ${current_price}
- **Change:** {percent_change}% (Today)
- **Volume:** {volume_str} (Avg: {avg_volume_str}) -> Volume Relative: {vol_relative}x
- **RSI (14):** {rsi}
- **ATR:** ${atr} (Use for Stop Loss distance)

### 2. TREND & LEVELS
- **SMA 20:** ${sma_20} | **SMA 50:** ${sma_50} | **SMA 200:** ${sma_200}
- **52W High:** ${year_high} | **52W Low:** ${year_low}
- **Nearest Support:** ${support_level}
- **Nearest Resistance:** ${resistance_level}

### 3. SQUEEZE METRICS (Crucial)
- **Short Interest:** {short_interest}%
- **Float:** {float_shares} shares
- **Days to Cover:** {days_to_cover}

### 4. NEWS CATALYSTS (Last 24h)
{news_headlines_formatted}

### 5. PAST LESSONS LEARNED (DO NOT REPEAT THESE MISTAKES):
{lessons_text}

---
### ⚡ YOUR TASK:
Based strictly on the "Logic Engines" defined in your System Prompt:
1. Determine if this is a Sympathy, Squeeze, Gamma, or Macro play.
2. If Volume Relative < 1.0 and no news -> HARD PASS (unless setup is perfect).
3. If RSI > 75 -> Look for potential blow-off top or pullback entry.
4. **Calculate specific numbers:**
   - Entry: Current price or pullback to ${sma_20}?
   - Stop Loss: Must be below ${support_level} or based on ATR (-2x ATR = ${atr * 2 if isinstance(atr, (int, float)) else 'N/A'}).
   - Target: Minimum 3x risk (Risk = Entry - Stop).

RETURN JSON ONLY.
"""
        return prompt

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _parse_price(self, value) -> Optional[float]:
        """
        Parse a price value from various formats.
        Handles: "$100", "100.50", "$100-$105" (returns average), "100", etc.
        """
        if value is None:
            return None

        # If already a number, return it
        if isinstance(value, (int, float)):
            return float(value)

        # Convert to string and clean
        price_str = str(value).strip()

        # Handle ranges like "$100-$105" or "100-105"
        if '-' in price_str and not price_str.startswith('-'):
            parts = price_str.split('-')
            if len(parts) == 2:
                try:
                    low = self._parse_price(parts[0])
                    high = self._parse_price(parts[1])
                    if low is not None and high is not None:
                        return (low + high) / 2
                except (ValueError, TypeError):
                    pass

        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[$,\s]', '', price_str)

        # Extract number (including decimals and negative)
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def _validate_trade_logic(self, trade_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        🛡️ THE KILL SWITCH - Validates trade logic after LLM response.

        Checks:
        1. Logical direction (BUY: Target > Entry > Stop, SELL: opposite)
        2. Risk/Reward ratio >= 2.5

        Modifies trade_json in place and returns it.
        """
        action = trade_json.get("action", "PASS").upper()

        # Skip validation for PASS actions
        if action == "PASS":
            return trade_json

        plan = trade_json.get("plan", {})
        reasoning = trade_json.get("reasoning", "")
        validation_notes = []

        # Extract prices
        entry = self._parse_price(plan.get("buy_zone"))
        targets = plan.get("targets", [])
        target = self._parse_price(targets[0]) if targets else None
        stop_loss = self._parse_price(plan.get("invalidation"))

        # If we can't parse prices, downgrade but don't fail
        if entry is None or target is None or stop_loss is None:
            validation_notes.append("⚠️ Could not parse all prices for validation")
            trade_json["reasoning"] = f"{reasoning}\n{' | '.join(validation_notes)}"
            return trade_json

        # ========== LOGICAL DIRECTION CHECK ==========
        direction_valid = True

        if action == "BUY":
            # For BUY: Target > Entry > Stop Loss
            if not (target > entry):
                direction_valid = False
                validation_notes.append(f"❌ MATH ERROR: Target (${target:.2f}) <= Entry (${entry:.2f})")
            if not (entry > stop_loss):
                direction_valid = False
                validation_notes.append(f"❌ MATH ERROR: Entry (${entry:.2f}) <= Stop (${stop_loss:.2f})")

        elif action == "SELL":
            # For SELL/SHORT: Target < Entry < Stop Loss
            if not (target < entry):
                direction_valid = False
                validation_notes.append(f"❌ MATH ERROR: Target (${target:.2f}) >= Entry (${entry:.2f})")
            if not (entry < stop_loss):
                direction_valid = False
                validation_notes.append(f"❌ MATH ERROR: Entry (${entry:.2f}) >= Stop (${stop_loss:.2f})")

        # If direction is invalid, force PASS
        if not direction_valid:
            trade_json["action"] = "PASS"
            trade_json["conviction"] = "LOW"
            trade_json["is_push_alert"] = False
            validation_notes.insert(0, "🚫 TRADE INVALIDATED - Illogical price direction")

        # ========== RISK/REWARD CALCULATION ==========
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)

        if risk > 0:
            rr_ratio = reward / risk
            trade_json["calculated_rr"] = round(rr_ratio, 2)

            # R/R must be at least 2.5 (buffer for AI rounding)
            if rr_ratio < 2.5:
                validation_notes.append(f"⚠️ R/R too low ({rr_ratio:.2f}:1) - Need minimum 2.5:1")

                # Downgrade conviction but don't necessarily PASS
                if trade_json.get("conviction") == "HIGH":
                    trade_json["conviction"] = "MED"
                elif trade_json.get("conviction") == "MED":
                    trade_json["conviction"] = "LOW"

                # If R/R is really bad (< 1.5), force PASS
                if rr_ratio < 1.5:
                    trade_json["action"] = "PASS"
                    trade_json["is_push_alert"] = False
                    validation_notes.append("🚫 R/R < 1.5 - FORCED PASS")
            else:
                validation_notes.append(f"✅ R/R Valid ({rr_ratio:.2f}:1)")
        else:
            validation_notes.append("⚠️ Risk = 0, cannot calculate R/R")
            trade_json["conviction"] = "LOW"

        # ========== ADD VALIDATION SUMMARY ==========
        trade_json["validation"] = {
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "risk": round(risk, 2),
            "reward": round(reward, 2),
            "rr_ratio": round(reward / risk, 2) if risk > 0 else 0,
            "passed": direction_valid and (reward / risk >= 2.5 if risk > 0 else False),
            "notes": validation_notes
        }

        # Append validation notes to reasoning
        if validation_notes:
            trade_json["reasoning"] = f"{reasoning}\n\n📋 **Validation:** {' | '.join(validation_notes)}"

        return trade_json

    def _call_ai_api(self, user_prompt: str, ticker: str = "???") -> str:
        """
        Call the AI API with:
        - Global rate limiting (min 5s between calls)
        - Single-flight locking (only 1 concurrent request)
        - Retry with exponential backoff + jitter for 429 errors
        """
        rate_limiter = get_rate_limiter(min_interval=5.0)
        single_flight = get_single_flight()

        # Acquire rate limit slot first
        wait_time = rate_limiter.acquire()
        if wait_time > 0:
            log_debug(f"Rate limiter waited {wait_time:.2f}s")

        # Acquire single-flight lock
        request_id = single_flight.acquire(ticker)

        try:
            log_info(f"[{request_id}] LLM Request: provider={self.provider}, ticker={ticker}")

            # Call the actual API with retry logic
            response_text = self._execute_api_call(user_prompt, request_id, ticker)

            log_ok(f"[{request_id}] LLM Response received ({len(response_text)} chars)")
            return response_text

        finally:
            # Always release the lock
            single_flight.release(request_id)

    @retry_with_backoff_jitter(max_retries=5, base_delay=5.0, max_delay=90.0, jitter_factor=0.2)
    def _execute_api_call(self, user_prompt: str, request_id: str = "???", ticker: str = "???") -> str:
        """
        Execute the actual API call to the LLM provider.
        Decorated with retry logic for 429 errors.
        """
        response_text = ""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": CIO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=CIO_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = response.content[0].text

        elif self.provider == "google":
            full_prompt = f"{CIO_SYSTEM_PROMPT}\n\n{user_prompt}"
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt
            )
            response_text = response.text

        return response_text

    def analyze_ticker(
        self,
        ticker: str,
        market_data: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Main analysis function with smart caching.

        Caching Strategy:
        - Returns cached result if:
          1. Cache entry exists and is < 60 minutes old
          2. Price has moved < 0.5% since cached analysis
        - Otherwise runs fresh analysis

        Args:
            ticker: Stock ticker symbol
            market_data: Optional pre-fetched market data
            skip_cache: If True, always run fresh analysis
        """
        log_divider(f"CIO ANALYSIS: {ticker}")
        log_ai(f"Analyzing {ticker}...")

        if not self.provider:
            log_error("No AI provider configured", include_trace=False)
            return {
                "error": "No AI provider configured",
                "bluf": "Analysis unavailable - configure API key",
                "action": "PASS",
                "ticker": ticker,
                "conviction": "LOW",
                "is_push_alert": False
        }

        # Fetch market data if not provided
        if not market_data:
            log_ai(f"Fetching market data for {ticker}...")
            market_data = self.fetch_market_data(ticker)

        price = market_data.get('price')
        vol_ratio = market_data.get('volume_ratio', 'N/A')
        log_debug(f"Data fetched: Price=${price}, Vol Ratio={vol_ratio}x")

        # =====================================================================
        # SMART CACHE CHECK
        # =====================================================================
        cache = get_analysis_cache()

        if not skip_cache and price and price > 0:
            cached_result = cache.get(ticker, price)

            if cached_result:
                # Cache hit! Return cached analysis with metadata
                cache_age = cached_result.get('_cache_age_minutes', 0)
                cache_price = cached_result.get('_cache_price', 0)
                cache_hits = cached_result.get('_cache_hits', 0)

                log_ok(f"📦 CACHE HIT for {ticker} (age: {cache_age:.0f}min, hits: {cache_hits})")

                # Update the BLUF to indicate cached result
                cached_result['bluf'] = f"[📦 Cached] {cached_result.get('bluf', '')}"
                cached_result['_source'] = 'cache'
                cached_result['raw_data'] = market_data  # Update with current market data

                return cached_result
            else:
                log_debug(f"Cache miss for {ticker} - running fresh analysis")
        elif skip_cache:
            log_debug(f"Cache skipped for {ticker} (skip_cache=True)")

        # =====================================================================
        # FRESH ANALYSIS
        # =====================================================================
        user_prompt = self._build_user_prompt(market_data)
        log_debug(f"Prompt built ({len(user_prompt)} chars)")

        try:
            log_ai(f"Calling {self.provider.upper()} API...")

            # Call API with rate limiting + single-flight + retry logic
            response_text = self._call_ai_api(user_prompt, ticker=ticker)

            # 🔍 CRITICAL: Print raw JSON for debugging
            log_ai_raw(response_text)

            # Parse JSON response
            result = self._parse_json_response(response_text)

            if result:
                log_ok(f"JSON parsed successfully")
                # 🛡️ KILL SWITCH: Validate trade logic before returning
                log_ai("Running Kill Switch validation...")
                result = self._validate_trade_logic(result)

                # Log validation result
                validation = result.get("validation", {})
                rr_ratio = validation.get("rr_ratio", 0)
                passed = validation.get("passed", False)
                action = result.get("action", "PASS")
                conviction = result.get("conviction", "LOW")

                log_validation(passed, f"R/R: {rr_ratio:.2f}:1 | Action: {action} | Conviction: {conviction}")
                log_trade(action, ticker, f"Conviction: {conviction}")

                # =====================================================================
                # CACHE THE RESULT
                # =====================================================================
                if price and price > 0:
                    cache.set(ticker, price, result)
                    log_debug(f"Cached analysis for {ticker} @ ${price:.2f}")

                result["raw_data"] = market_data
                result["_source"] = "fresh"
                return result
            else:
                log_error(f"Failed to parse JSON from AI response", include_trace=False)
                log_debug(f"Raw response: {response_text[:500]}")
                return {
                    "error": "Failed to parse AI response",
                    "raw_response": response_text,
                    "bluf": "Analysis parsing failed",
                    "action": "PASS",
                    "ticker": ticker,
                    "conviction": "LOW",
                    "is_push_alert": False
                }

        except Exception as e:
            log_error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "bluf": f"Analysis failed: {e}",
                "action": "PASS",
                "ticker": ticker,
                "conviction": "LOW",
                "is_push_alert": False
            }

    def format_hebrew_response(self, analysis: Dict[str, Any]) -> str:
        """
        Format the JSON analysis into Hebrew Telegram message.
        """

        ticker = analysis.get("ticker", "???")
        bluf = analysis.get("bluf", "אין סיכום")
        logic = analysis.get("logic_engine", "N/A")
        action = analysis.get("action", "PASS")
        conviction = analysis.get("conviction", "LOW")
        is_alert = analysis.get("is_push_alert", False)
        reasoning = analysis.get("reasoning", "")
        plan = analysis.get("plan", {})
        validation = analysis.get("validation", {})

        # Cache info
        is_cached = analysis.get("_cached", False)
        cache_age = analysis.get("_cache_age_minutes", 0)
        source = analysis.get("_source", "fresh")

        # Action emoji
        action_emoji = {
            "BUY": "🟢 קנייה",
            "SELL": "🔴 מכירה",
            "PASS": "⏸️ המתנה"
        }.get(action, "⚠️ לא ידוע")

        # Conviction emoji
        conv_emoji = {
            "HIGH": "🔥",
            "MED": "⚡",
            "LOW": "💤"
        }.get(conviction, "❓")

        # Validation status
        validation_status = ""
        if validation:
            rr = validation.get("rr_ratio", 0)
            passed = validation.get("passed", False)
            risk = validation.get("risk", 0)
            reward = validation.get("reward", 0)

            status_icon = "✅" if passed else "⚠️"
            validation_status = f"""
🛡️ **Kill Switch:** {status_icon}
• Risk: ${risk:.2f} | Reward: ${reward:.2f}
• R/R: **{rr:.2f}:1** {'✅' if rr >= 2.5 else '⚠️'}"""

        # Cache indicator
        cache_badge = ""
        if is_cached:
            cache_badge = f"📦 *[Cached: {cache_age:.0f}m ago]*\n"

        # Build message
        header = "🚨 **[PUSH ALERT]** 🚨\n" if is_alert else ""
        header = cache_badge + header

        msg = f"""{header}📊 **ניתוח CIO: {ticker}**
━━━━━━━━━━━━━━━━━━━━

🎯 **BLUF:** {bluf}

🧠 **Logic Engine:** {logic}

{action_emoji} **פעולה:** {action}
{conv_emoji} **ביטחון:** {conviction}

📋 **תוכנית מסחר:**
• אזור כניסה: {plan.get('buy_zone', 'N/A')}
• יעד 1: {plan.get('targets', ['N/A'])[0] if plan.get('targets') else 'N/A'}
• יעד 2: {plan.get('targets', ['N/A', 'N/A'])[1] if len(plan.get('targets', [])) > 1 else 'N/A'}
• סטופ: {plan.get('invalidation', 'N/A')}
{validation_status}

💡 **נימוק:**
{reasoning}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return msg


# Singleton instance for easy import
_brain_instance = None

def get_brain() -> AIBrain:
    """Get singleton AIBrain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AIBrain()
    return _brain_instance


if __name__ == "__main__":
    # Test the AI Brain
    brain = get_brain()
    print(f"Using provider: {brain.provider}")

    result = brain.analyze_ticker("NVDA")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "="*50 + "\n")
    print(brain.format_hebrew_response(result))
