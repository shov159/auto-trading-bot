"""
News Brain - Financial News Intelligence Module
================================================
Fetches and analyzes financial news using FMP API + AI sentiment analysis.
"""
import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from src.logger import log_info, log_warn, log_error, log_ok, log_debug, log_ai

# =============================================================================
# CONFIGURATION
# =============================================================================
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# AI Provider (reuse from ai_brain)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# =============================================================================
# NEWS SENTIMENT PROMPT
# =============================================================================
NEWS_SENTIMENT_PROMPT = """You are a senior Wall Street analyst with 20+ years of experience.
Your job is to analyze financial news and rate its impact on a stock's price.

RATING SCALE (-10 to +10):
- **+10:** Extremely Bullish (acquisition, massive earnings beat, breakthrough product)
- **+7 to +9:** Very Bullish (strong guidance, major contract win)
- **+4 to +6:** Moderately Bullish (positive analyst upgrade, good quarter)
- **+1 to +3:** Slightly Bullish (minor positive news)
- **0:** Neutral (no significant impact)
- **-1 to -3:** Slightly Bearish (minor concerns, small miss)
- **-4 to -6:** Moderately Bearish (downgrade, guidance cut)
- **-7 to -9:** Very Bearish (lawsuit, major miss, executive departure)
- **-10:** Extremely Bearish (fraud, bankruptcy, delisting)

ANALYSIS GUIDELINES:
1. Consider IMMEDIATE market reaction (1-5 days)
2. Factor in whether the news is priced in
3. Consider sector/market context
4. Be realistic - most news is -3 to +3

OUTPUT JSON ONLY:
{
    "score": <number -10 to +10>,
    "sentiment": "<BULLISH|BEARISH|NEUTRAL>",
    "impact_summary": "<1 sentence explaining the impact>",
    "trade_recommendation": "<BUY|SELL|HOLD|WATCH>",
    "time_horizon": "<immediate|short_term|long_term>",
    "confidence": "<HIGH|MEDIUM|LOW>"
}
"""


# =============================================================================
# FMP API FUNCTIONS
# =============================================================================

class NewsBrain:
    """Financial News Intelligence Engine."""
    
    def __init__(self):
        self.fmp_key = FMP_API_KEY
        self.ai_provider = self._detect_ai_provider()
        
        if not self.fmp_key:
            log_warn("FMP_API_KEY not set - news features will be limited")
        else:
            log_info("NewsBrain initialized with FMP API")
    
    def _detect_ai_provider(self) -> Optional[str]:
        """Detect available AI provider."""
        if GOOGLE_API_KEY:
            return "google"
        elif OPENAI_API_KEY:
            return "openai"
        return None
    
    def fetch_stock_news(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch news for a specific stock ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'TSLA')
            limit: Number of news items to fetch
            
        Returns:
            List of news items with title, text, url, publishedDate
        """
        if not self.fmp_key:
            log_warn("FMP API key not configured")
            return self._fallback_news(ticker)
        
        try:
            url = f"{FMP_BASE_URL}/stock_news"
            params = {
                "tickers": ticker.upper(),
                "limit": limit,
                "apikey": self.fmp_key
            }
            
            log_debug(f"Fetching news for {ticker}...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news = response.json()
                log_ok(f"Fetched {len(news)} news items for {ticker}")
                return news
            else:
                log_error(f"FMP API error: {response.status_code}")
                return self._fallback_news(ticker)
                
        except Exception as e:
            log_error(f"Error fetching stock news: {e}")
            return self._fallback_news(ticker)
    
    def fetch_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch general market/macro news.
        
        Returns:
            List of general market news items
        """
        if not self.fmp_key:
            log_warn("FMP API key not configured")
            return []
        
        try:
            url = f"{FMP_BASE_URL}/fmp/articles"
            params = {
                "page": 0,
                "size": limit,
                "apikey": self.fmp_key
            }
            
            log_debug("Fetching market news...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news = data.get("content", [])
                log_ok(f"Fetched {len(news)} market news items")
                return news
            else:
                log_error(f"FMP API error: {response.status_code}")
                return []
                
        except Exception as e:
            log_error(f"Error fetching market news: {e}")
            return []
    
    def fetch_press_releases(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Fetch official press releases for a ticker."""
        if not self.fmp_key:
            return []
        
        try:
            url = f"{FMP_BASE_URL}/press-releases/{ticker.upper()}"
            params = {
                "limit": limit,
                "apikey": self.fmp_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return []
            
        except Exception as e:
            log_error(f"Error fetching press releases: {e}")
            return []
    
    def fetch_earnings_surprises(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch recent earnings surprises for a ticker."""
        if not self.fmp_key:
            return []
        
        try:
            url = f"{FMP_BASE_URL}/earnings-surprises/{ticker.upper()}"
            params = {
                "apikey": self.fmp_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data[:3] if data else []  # Last 3 quarters
            return []
            
        except Exception as e:
            log_error(f"Error fetching earnings: {e}")
            return []
    
    def _fallback_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fallback using yfinance when FMP is unavailable."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Convert to FMP format
            formatted = []
            for item in news[:5]:
                formatted.append({
                    "title": item.get("title", ""),
                    "text": item.get("title", ""),  # yfinance doesn't have full text
                    "url": item.get("link", ""),
                    "publishedDate": datetime.fromtimestamp(
                        item.get("providerPublishTime", time.time())
                    ).isoformat(),
                    "site": item.get("publisher", ""),
                    "symbol": ticker
                })
            
            log_info(f"Using yfinance fallback: {len(formatted)} news items")
            return formatted
            
        except Exception as e:
            log_error(f"Fallback news fetch failed: {e}")
            return []
    
    # =========================================================================
    # AI SENTIMENT ANALYSIS
    # =========================================================================
    
    def analyze_news_sentiment(
        self, 
        news_item: Dict[str, Any],
        ticker: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze news sentiment using AI.
        
        Args:
            news_item: News item dict with 'title' and optionally 'text'
            ticker: Stock ticker for context
            
        Returns:
            Sentiment analysis dict with score, summary, recommendation
        """
        if not self.ai_provider:
            return self._basic_sentiment(news_item)
        
        title = news_item.get("title", "")
        text = news_item.get("text", "")[:500]  # Limit text length
        source = news_item.get("site", "Unknown")
        date = news_item.get("publishedDate", "")
        
        user_prompt = f"""ANALYZE THIS NEWS FOR {ticker if ticker else 'the market'}:

**Headline:** {title}

**Summary:** {text}

**Source:** {source}
**Date:** {date}

Provide your analysis as JSON only."""

        try:
            if self.ai_provider == "google":
                return self._analyze_with_gemini(user_prompt)
            elif self.ai_provider == "openai":
                return self._analyze_with_openai(user_prompt)
            else:
                return self._basic_sentiment(news_item)
                
        except Exception as e:
            log_error(f"AI sentiment analysis failed: {e}")
            return self._basic_sentiment(news_item)
    
    def _analyze_with_gemini(self, user_prompt: str) -> Dict[str, Any]:
        """Analyze using Google Gemini."""
        try:
            from google import genai
            
            client = genai.Client(api_key=GOOGLE_API_KEY)
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"role": "user", "parts": [{"text": NEWS_SENTIMENT_PROMPT}]},
                    {"role": "model", "parts": [{"text": "I understand. I'll analyze news and provide JSON with score, sentiment, impact_summary, trade_recommendation, time_horizon, and confidence."}]},
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ]
            )
            
            result_text = response.text.strip()
            
            # Parse JSON from response
            return self._parse_sentiment_json(result_text)
            
        except Exception as e:
            log_error(f"Gemini analysis error: {e}")
            raise
    
    def _analyze_with_openai(self, user_prompt: str) -> Dict[str, Any]:
        """Analyze using OpenAI."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": NEWS_SENTIMENT_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_sentiment_json(result_text)
            
        except Exception as e:
            log_error(f"OpenAI analysis error: {e}")
            raise
    
    def _parse_sentiment_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        try:
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
            
        except json.JSONDecodeError:
            # Try to extract JSON object
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            
            log_warn(f"Failed to parse sentiment JSON: {text[:200]}")
            return {
                "score": 0,
                "sentiment": "NEUTRAL",
                "impact_summary": "Unable to analyze",
                "trade_recommendation": "HOLD",
                "time_horizon": "unknown",
                "confidence": "LOW"
            }
    
    def _basic_sentiment(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """Basic keyword-based sentiment when AI is unavailable."""
        title = news_item.get("title", "").lower()
        
        # Positive keywords
        positive = ["surge", "soar", "beat", "record", "upgrade", "buy", 
                   "growth", "profit", "bullish", "rally", "breakout"]
        # Negative keywords
        negative = ["crash", "plunge", "miss", "downgrade", "sell", "loss",
                   "bearish", "decline", "recall", "lawsuit", "fraud"]
        
        pos_count = sum(1 for word in positive if word in title)
        neg_count = sum(1 for word in negative if word in title)
        
        if pos_count > neg_count:
            score = min(pos_count * 2, 6)
            sentiment = "BULLISH"
            rec = "WATCH"
        elif neg_count > pos_count:
            score = max(-neg_count * 2, -6)
            sentiment = "BEARISH"
            rec = "WATCH"
        else:
            score = 0
            sentiment = "NEUTRAL"
            rec = "HOLD"
        
        return {
            "score": score,
            "sentiment": sentiment,
            "impact_summary": "Basic keyword analysis (AI unavailable)",
            "trade_recommendation": rec,
            "time_horizon": "unknown",
            "confidence": "LOW"
        }
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    def get_full_news_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive news analysis for a ticker.
        
        Returns:
            Dict with news items, sentiment scores, and overall assessment
        """
        log_ai(f"Running full news analysis for {ticker}...")
        
        # Fetch news
        news_items = self.fetch_stock_news(ticker, limit=5)
        
        if not news_items:
            return {
                "ticker": ticker,
                "news_count": 0,
                "overall_sentiment": "NEUTRAL",
                "overall_score": 0,
                "analyzed_items": [],
                "summary": "No recent news found"
            }
        
        # Analyze each item
        analyzed = []
        total_score = 0
        
        for item in news_items[:3]:  # Analyze top 3
            log_debug(f"Analyzing: {item.get('title', '')[:50]}...")
            
            sentiment = self.analyze_news_sentiment(item, ticker)
            
            analyzed.append({
                "title": item.get("title", ""),
                "date": item.get("publishedDate", ""),
                "source": item.get("site", ""),
                "url": item.get("url", ""),
                **sentiment
            })
            
            total_score += sentiment.get("score", 0)
            
            # Rate limit between analyses
            time.sleep(1)
        
        # Calculate overall
        avg_score = total_score / len(analyzed) if analyzed else 0
        
        if avg_score >= 3:
            overall = "BULLISH"
        elif avg_score <= -3:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"
        
        return {
            "ticker": ticker,
            "news_count": len(news_items),
            "analyzed_count": len(analyzed),
            "overall_sentiment": overall,
            "overall_score": round(avg_score, 1),
            "analyzed_items": analyzed,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_market_sentiment(self, limit: int = 5) -> Dict[str, Any]:
        """
        Get overall market sentiment from macro news.
        
        Returns:
            Dict with market mood assessment
        """
        log_ai("Analyzing market sentiment...")
        
        news_items = self.fetch_market_news(limit=limit)
        
        if not news_items:
            return {
                "mood": "NEUTRAL",
                "score": 0,
                "headlines": [],
                "summary": "No market news available"
            }
        
        # Analyze top items
        analyzed = []
        total_score = 0
        
        for item in news_items[:3]:
            title = item.get("title", "")
            if not title:
                continue
                
            sentiment = self.analyze_news_sentiment(item, "MARKET")
            
            analyzed.append({
                "title": title,
                "score": sentiment.get("score", 0),
                "sentiment": sentiment.get("sentiment", "NEUTRAL")
            })
            
            total_score += sentiment.get("score", 0)
            time.sleep(1)
        
        avg_score = total_score / len(analyzed) if analyzed else 0
        
        if avg_score >= 2:
            mood = "RISK-ON 🟢"
        elif avg_score <= -2:
            mood = "RISK-OFF 🔴"
        else:
            mood = "MIXED 🟡"
        
        return {
            "mood": mood,
            "score": round(avg_score, 1),
            "headlines": analyzed,
            "timestamp": datetime.now().isoformat()
        }
    
    # =========================================================================
    # TELEGRAM FORMATTING
    # =========================================================================
    
    def format_news_for_telegram(self, analysis: Dict[str, Any]) -> str:
        """Format news analysis for Telegram message."""
        ticker = analysis.get("ticker", "")
        overall = analysis.get("overall_sentiment", "NEUTRAL")
        score = analysis.get("overall_score", 0)
        items = analysis.get("analyzed_items", [])
        
        # Overall sentiment emoji
        if score >= 3:
            emoji = "🟢"
            mood = "Bullish"
        elif score <= -3:
            emoji = "🔴"
            mood = "Bearish"
        elif score > 0:
            emoji = "🟡"
            mood = "Slightly Bullish"
        elif score < 0:
            emoji = "🟠"
            mood = "Slightly Bearish"
        else:
            emoji = "⚪"
            mood = "Neutral"
        
        msg = f"""📰 **News Analysis: {ticker}**
━━━━━━━━━━━━━━━━━━━━

{emoji} **Overall Sentiment:** {mood} ({score:+.1f})

"""
        
        for i, item in enumerate(items, 1):
            title = item.get("title", "")[:80]
            item_score = item.get("score", 0)
            rec = item.get("trade_recommendation", "HOLD")
            summary = item.get("impact_summary", "")[:100]
            
            # Score emoji
            if item_score >= 3:
                score_emoji = "🟢"
            elif item_score <= -3:
                score_emoji = "🔴"
            else:
                score_emoji = "🟡"
            
            msg += f"""**{i}. {title}**
   {score_emoji} Score: **{item_score:+d}** | Rec: `{rec}`
   _{summary}_

"""
        
        # Add timestamp
        msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return msg
    
    def format_macro_for_telegram(self, analysis: Dict[str, Any]) -> str:
        """Format market sentiment for Telegram."""
        mood = analysis.get("mood", "NEUTRAL")
        score = analysis.get("score", 0)
        headlines = analysis.get("headlines", [])
        
        msg = f"""🌍 **Market Macro Sentiment**
━━━━━━━━━━━━━━━━━━━━

**Market Mood:** {mood}
**Sentiment Score:** {score:+.1f}

📰 **Top Headlines:**
"""
        
        for item in headlines:
            title = item.get("title", "")[:70]
            item_score = item.get("score", 0)
            sentiment = item.get("sentiment", "NEUTRAL")
            
            emoji = "🟢" if item_score > 0 else "🔴" if item_score < 0 else "⚪"
            msg += f"\n{emoji} {title}\n   Score: {item_score:+d} ({sentiment})\n"
        
        msg += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return msg


# =============================================================================
# SINGLETON
# =============================================================================
_news_brain_instance: Optional[NewsBrain] = None


def get_news_brain() -> NewsBrain:
    """Get or create NewsBrain instance."""
    global _news_brain_instance
    if _news_brain_instance is None:
        _news_brain_instance = NewsBrain()
    return _news_brain_instance


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def analyze_ticker_news(ticker: str) -> Dict[str, Any]:
    """Quick function to analyze news for a ticker."""
    return get_news_brain().get_full_news_analysis(ticker)


def get_market_mood() -> Dict[str, Any]:
    """Quick function to get market sentiment."""
    return get_news_brain().get_market_sentiment()


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    brain = NewsBrain()
    
    # Test stock news
    print("\n=== Testing Stock News ===")
    analysis = brain.get_full_news_analysis("TSLA")
    print(brain.format_news_for_telegram(analysis))
    
    # Test market news
    print("\n=== Testing Market News ===")
    macro = brain.get_market_sentiment()
    print(brain.format_macro_for_telegram(macro))

