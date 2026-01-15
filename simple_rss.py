import feedparser

def test_feed():
    url = "https://finance.yahoo.com/news/rssindex"
    print(f"Connecting to {url}...")
    feed = feedparser.parse(url)
    
    if feed.entries:
        print(f"Success! Found {len(feed.entries)} items.")
        print("First headline:", feed.entries[0].title)
    else:
        print("Failed to get entries.")

if __name__ == "__main__":
    test_feed()

