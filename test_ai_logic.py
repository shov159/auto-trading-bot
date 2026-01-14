import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

# Load environment variables
load_dotenv()

# Mock Strategy Class to mimic the real one structure
class MockStrategy:
    def __init__(self):
        self.symbol = "SPY"
        self.news_client = None
        self.client = None
        
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            self.news_client = NewsClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
        
        if GOOGLE_API_KEY:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def get_sentiment(self):
        print("\n--- 1. Fetching News ---")
        news_headlines = []
        if self.news_client:
            try:
                request = NewsRequest(symbols=self.symbol, limit=5)
                # Raw response from Alpaca
                response = self.news_client.get_news(request)
                
                # Logic to handle tuple return: ('data', {'news': [...]})
                if isinstance(response, tuple):
                    # Look for the dict in the tuple
                    data_dict = next((item for item in response if isinstance(item, dict)), None)
                    if data_dict and 'news' in data_dict:
                        news_items = data_dict['news']
                    else:
                        news_items = []
                elif hasattr(response, 'news'):
                    news_items = response.news
                else:
                    news_items = response

                # Process items
                processed_headlines = []
                for item in news_items:
                    # Item can be a dict (from raw tuple) or an object (from SDK object)
                    if isinstance(item, dict):
                        processed_headlines.append(item.get('headline', ''))
                    elif hasattr(item, 'headline'):
                        processed_headlines.append(item.headline)
                    else:
                        processed_headlines.append(str(item))

                news_headlines = processed_headlines
                
                print(f"Headlines fetched for {self.symbol}:")
                for h in news_headlines:
                    print(f"  - {h}")
            except Exception as e:
                print(f"Error fetching news: {e}")

        if not news_headlines:
            print("No news found, using fallback mock data.")
            news_headlines = ["Mock News: SPY looks bullish", "Mock News: Tech rally continues"]

        print("\n--- 2. Analyzing Sentiment with Gemini ---")
        prompt = f"""
        You are a financial analyst. Analyze the sentiment of the following news headlines for {self.symbol}:
        {news_headlines}

        Return a single word: BUY, SELL, or HOLD.
        """

        if not self.client:
            print("No Google API Key. Returning Mock BUY.")
            return "BUY"

        try:
            # FIX: Use 'gemini-flash-latest'
            response = self.client.models.generate_content(
                model='gemini-flash-latest', contents=prompt
            )
            content = response.text.strip().upper()
            print(f"Google Gemini Raw Response: {content}")
            return content
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return "ERROR"

if __name__ == "__main__":
    print("Starting Dry Run Logic Test...")
    strat = MockStrategy()
    decision = strat.get_sentiment()
    print(f"\n--- Final Decision: {decision} ---")
