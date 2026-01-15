import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.news_scout import NewsScout

def test_headlines():
    print("Initializing NewsScout...")
    scout = NewsScout()
    
    print("\nFetching latest headlines...")
    headlines = scout.fetch_headlines(limit_per_feed=5)
    
    print("\n--- Latest Financial News ---")
    for i, h in enumerate(headlines, 1):
        print(f"{i}. {h}")
        
    print(f"\nTotal Headlines: {len(headlines)}")

if __name__ == "__main__":
    test_headlines()

