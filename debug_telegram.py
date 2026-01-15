import os
import requests
from dotenv import load_dotenv

def test_telegram_connection():
    # 1. Load Env
    load_dotenv()
    
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print("--- Telegram Diagnostics ---")
    print(f"TELEGRAM_TOKEN: {'Present' if token else 'Missing'}")
    print(f"TELEGRAM_CHAT_ID: {'Present' if chat_id else 'Missing'}")
    
    if not token or not chat_id:
        print("[ERR] Cannot proceed. Missing credentials.")
        return

    # 2. Verify Token (getMe)
    print("\n[Step 1] Verifying Bot Token...")
    try:
        url_me = f"https://api.telegram.org/bot{token}/getMe"
        resp_me = requests.get(url_me, timeout=10)
        print(f"Status Code: {resp_me.status_code}")
        print(f"Response: {resp_me.json()}")
        
        if resp_me.status_code != 200:
            print("[ERR] Bot Token appears invalid.")
            return
        else:
            bot_name = resp_me.json().get('result', {}).get('first_name', 'Unknown')
            print(f"[OK] Bot Token Valid! Connected as: {bot_name}")
            
    except Exception as e:
        print(f"[ERR] Network Error calling getMe: {e}")
        return

    # 3. Verify Message Sending
    print("\n[Step 2] Sending Test Message...")
    try:
        url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "🔍 Debug Test: Connectivity Verified!"
        }
        resp_msg = requests.post(url_msg, json=payload, timeout=10)
        print(f"Status Code: {resp_msg.status_code}")
        print(f"Response: {resp_msg.json()}")
        
        if resp_msg.status_code == 200:
            print("[OK] Message Sent Successfully!")
        else:
            print("[ERR] Message Send Failed. Check Chat ID.")
            
    except Exception as e:
        print(f"[ERR] Network Error sending message: {e}")

if __name__ == "__main__":
    test_telegram_connection()

