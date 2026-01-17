"""News Scout - Multi-Category RSS Monitor"""
import feedparser
import os
import re
import yfinance as yf
from google import genai
from dotenv import load_dotenv

load_dotenv()

class NewsScout:
    def __init__(self):
        self.last_scores = {
            "us_markets": 0.0,
            "crypto": 0.0,
            "global_macro": 0.0
        }
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Gemini Client Init Error: {e}")
                self.gemini_client = None
        else:
            self.gemini_client = None
        
        # RSS Feed Categories
        self.feeds = {
            "crypto": [
                "https://www.coindesk.com/arc/outboundfeeds/rss/"
            ],
            "global_macro": [
                "https://www.reuters.com/finance/rss",
                "https://feeds.bloomberg.com/markets/news.rss"
            ]
        }
    
    def fetch_category_headlines(self, category):
        """Fetch headlines for a specific category"""
        headlines = []
        
        # USE YFINANCE FOR US MARKETS (Real Headlines)
        if category == "us_markets":
            try:
                # Fetch SPY news as proxy for US Market
                spy = yf.Ticker("SPY")
                news_items = spy.news
                if news_items:
                    # Get top 5 headlines
                    # Ensure safe key access
                    headlines = []
                    for item in news_items[:5]:
                        # Debug structure if needed (for user request)
                        # print(item) 
                        title = item.get('title')
                        if not title:
                            # Try 'headline' or just skip
                            title = item.get('headline')
                        
                        if title:
                            headlines.append(title)
                            
                    if not headlines:
                         headlines = ["No specific headlines found."]
                         
            except Exception as e:
                print(f"YFinance News Error: {e}")
                headlines = ["Error fetching US market news"]
                
        # USE RSS FOR OTHERS
        else:
            for url in self.feeds.get(category, []):
                try:
                    feed = feedparser.parse(url)
                    headlines.extend([entry.title for entry in feed.entries[:5]])
                except:
                    continue
                    
        return headlines[:10]  # Max 10 per category
    
    def analyze_category(self, category, headlines):
        """Analyze sentiment for a category"""
        if not headlines or not self.gemini_client:
            return 0.0, "No data/API"
        
        text_block = "\n".join(headlines)
        prompt = f"Analyze these {category.replace('_', ' ')} headlines. Give ONLY a numerical sentiment score (-1.0 to 1.0) and a brief 1-sentence insight.\n\n{text_block}"
        
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt
            )
            text = response.text
            match = re.search(r'[-+]?[0-9]*\.?[0-9]+', text)
            score = float(match.group()) if match else 0.0
            return score, text
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def get_regime(self):
        """Returns the current market sentiment regime scores without checking thresholds."""
        from datetime import datetime
        
        regime = {
            "overall": 0.0,
            "us_markets": self.last_scores["us_markets"],
            "crypto": self.last_scores["crypto"],
            "global_macro": self.last_scores["global_macro"],
            "timestamp_utc": datetime.utcnow().isoformat()
        }
        
        # Calculate scores if not already done in this cycle? 
        # Actually, get_market_brief updates self.last_scores only if delta > 0.15.
        # But for trading logic, we might want the FRESH score every time, not just alerts.
        # So we should probably fetch and analyze here, OR decouple fetching from alerting.
        
        # Refactoring to separate analysis from alerting:
        scores = {}
        for category in ["us_markets", "crypto", "global_macro"]:
            headlines = self.fetch_category_headlines(category)
            score, _ = self.analyze_category(category, headlines)
            scores[category] = score
            # Update internal state silently so get_market_brief doesn't re-trigger on same data?
            # Or better: let get_regime be the source of truth.
            self.last_scores[category] = score # Update state
            
        regime["us_markets"] = scores["us_markets"]
        regime["crypto"] = scores["crypto"]
        regime["global_macro"] = scores["global_macro"]
        
        # Simple average for overall
        total = sum(scores.values())
        regime["overall"] = total / 3 if scores else 0.0
        
        return regime

    def get_market_brief(self):
        """Main method: Returns dashboard if any category changed significantly"""
        report_lines = []
        alerts = []
        
        # Use current state (updated via get_regime or lazily here)
        # To ensure we have fresh data if get_regime wasn't called:
        
        current_scores = {}
        
        for category in ["us_markets", "crypto", "global_macro"]:
            headlines = self.fetch_category_headlines(category)
            score, summary = self.analyze_category(category, headlines)
            current_scores[category] = score
            
            # Check for significant change against PREVIOUS state
            # Logic issue: if get_regime just updated self.last_scores, delta is 0.
            # We need to manage state carefully.
            
            # Simplified for Task 1: get_market_brief executes the fetch/analyze loop
            # and returns text. It also updates internal state.
            
            delta = abs(score - self.last_scores[category])
            
            if delta >= 0.15:
                self.last_scores[category] = score
                alerts.append(category)
            
            # Build report line
            emoji = "🇺🇸" if category == "us_markets" else "₿" if category == "crypto" else "🌍"
            mood = "חיובי" if score > 0.2 else "שלילי" if score < -0.2 else "ניטרלי"
            report_lines.append(f"{emoji} **{category.replace('_', ' ').title()}:** {score:+.2f} ({mood})")
        
        # Only send if at least one category changed
        if alerts:
            header = "🚨 **עדכון חדשות מתפרץ!**\n"
            body = "\n".join(report_lines)
            footer = f"\n\n🔔 שינויים ב: {', '.join(alerts)}"
            return header + body + footer
        
        return None
