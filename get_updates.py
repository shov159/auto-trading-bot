import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    print("Error: TELEGRAM_TOKEN not found in .env")
    exit(1)

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
print(f"Checking updates for bot token starting with: {TOKEN[:5]}...")

try:
    response = requests.get(url)
    print("Response JSON:")
    print(response.json())
except Exception as e:
    print(e)

