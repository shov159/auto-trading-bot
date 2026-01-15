import feedparser
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def analyze_market():
    # 1. Fetch Headlines
    url = "https://finance.yahoo.com/news/rssindex"
    feed = feedparser.parse(url)
    
    # Take top 10 headlines only
    headlines = [entry.title for entry in feed.entries[:10]]
    text_block = "\n".join(headlines)
    
    print("--- Top 10 Headlines ---")
    print(text_block)
    print("-" * 20)

    # 2. Ask Gemini
    # Note: Using 'gemini-flash-latest' as it's often the recommended alias for speed/availability in your region
    # If 'gemini-pro' fails, we can switch to 'gemini-1.5-flash' or similar.
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = f"Analyze these financial headlines. Give me a market sentiment score (-1 to 1) and a 1-sentence summary.\n\n{text_block}"
    
    print("Asking Gemini...")
    try:
        response = model.generate_content(prompt)
        print("\n--- Gemini Analysis ---")
        print(response.text)
    except Exception as e:
        print(f"\n[Error] Gemini failed: {e}")

if __name__ == "__main__":
    analyze_market()

