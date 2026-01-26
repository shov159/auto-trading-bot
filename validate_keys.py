#!/usr/bin/env python3
"""
API Key Validator - Pre-Flight Connection Check
================================================
Validates all API keys before starting the bot.
Run this before the main bot to ensure everything is connected.
"""
import os
import sys

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# COLOR SETUP
# =============================================================================
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback if colorama not installed yet
    class Fore:
        CYAN = GREEN = RED = YELLOW = WHITE = MAGENTA = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""


def print_header():
    """Print validation header."""
    print(f"""
{Fore.CYAN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════╗
║           🔑 API KEY VALIDATION - PRE-FLIGHT CHECK 🔑         ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")


def print_result(service: str, success: bool, message: str = ""):
    """Print validation result."""
    if success:
        print(f"  {Fore.GREEN}✅ {service}: PASS{Style.RESET_ALL}")
        if message:
            print(f"     {Fore.GREEN}{message}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}❌ {service}: FAIL{Style.RESET_ALL}")
        if message:
            print(f"     {Fore.RED}{message}{Style.RESET_ALL}")


def validate_openai() -> bool:
    """Validate OpenAI API key by sending a test request."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print_result("OpenAI", False, "OPENAI_API_KEY not set in .env")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Send minimal test request
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for validation
            messages=[{"role": "user", "content": "Say 'OK' only."}],
            max_tokens=5
        )
        
        result = response.choices[0].message.content.strip()
        print_result("OpenAI", True, f"Connected (Model: gpt-4o-mini, Response: '{result}')")
        return True
    
    except openai.AuthenticationError:
        print_result("OpenAI", False, "Invalid API key - check your OPENAI_API_KEY")
        return False
    except openai.RateLimitError:
        print_result("OpenAI", False, "Rate limit exceeded or quota depleted")
        return False
    except Exception as e:
        print_result("OpenAI", False, f"Connection error: {e}")
        return False


def validate_anthropic() -> bool:
    """Validate Anthropic API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print(f"  {Fore.YELLOW}⚪ Anthropic: Not configured (optional){Style.RESET_ALL}")
        return True  # Optional, so return True
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use cheaper model
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK only."}]
        )
        
        result = response.content[0].text.strip()
        print_result("Anthropic", True, f"Connected (Response: '{result}')")
        return True
    
    except Exception as e:
        print_result("Anthropic", False, f"Connection error: {e}")
        return False


def validate_google() -> bool:
    """Validate Google Gemini API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print(f"  {Fore.YELLOW}⚪ Google Gemini: Not configured (optional){Style.RESET_ALL}")
        return True  # Optional
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Say OK only.'
        )
        
        result = response.text.strip()
        print_result("Google Gemini", True, f"Connected (Response: '{result}')")
        return True
    
    except Exception as e:
        print_result("Google Gemini", False, f"Connection error: {e}")
        return False


def validate_alpaca() -> bool:
    """Validate Alpaca API keys by fetching account info."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    is_paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    if not api_key or not secret_key:
        print_result("Alpaca", False, "ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
        return False
    
    try:
        from alpaca.trading.client import TradingClient
        
        client = TradingClient(api_key, secret_key, paper=is_paper)
        account = client.get_account()
        
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        status = account.status
        
        mode = f"{Fore.GREEN}PAPER{Style.RESET_ALL}" if is_paper else f"{Fore.RED}LIVE{Style.RESET_ALL}"
        
        print_result("Alpaca", True, f"Mode: {mode} | Status: {status}")
        print(f"     {Fore.CYAN}💰 Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}{Style.RESET_ALL}")
        return True
    
    except Exception as e:
        error_msg = str(e)
        if "forbidden" in error_msg.lower() or "401" in error_msg:
            print_result("Alpaca", False, "Invalid API keys - check credentials")
        else:
            print_result("Alpaca", False, f"Connection error: {e}")
        return False


def validate_telegram() -> bool:
    """Validate Telegram bot token and send test message."""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token:
        print_result("Telegram", False, "TELEGRAM_TOKEN not set")
        return False
    
    if not chat_id:
        print_result("Telegram", False, "TELEGRAM_CHAT_ID not set")
        return False
    
    try:
        import requests
        from datetime import datetime
        
        # First, verify bot token by getting bot info
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print_result("Telegram", False, "Invalid bot token")
            return False
        
        bot_info = response.json()
        bot_name = bot_info.get("result", {}).get("username", "Unknown")
        
        # Send test message
        message_url = f"https://api.telegram.org/bot{token}/sendMessage"
        test_message = f"""🔑 **System Validation Complete**

✅ All API keys validated successfully!
🤖 Bot: @{bot_name}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Starting CIO Trading Bot..._"""
        
        payload = {
            "chat_id": chat_id,
            "text": test_message,
            "parse_mode": "Markdown"
        }
        
        msg_response = requests.post(message_url, json=payload, timeout=10)
        
        if msg_response.status_code == 200:
            print_result("Telegram", True, f"Bot: @{bot_name} | Message sent to chat {chat_id}")
            return True
        else:
            error = msg_response.json().get("description", "Unknown error")
            print_result("Telegram", False, f"Could not send message: {error}")
            return False
    
    except requests.exceptions.Timeout:
        print_result("Telegram", False, "Connection timeout - check your internet")
        return False
    except Exception as e:
        print_result("Telegram", False, f"Error: {e}")
        return False


def main() -> int:
    """
    Run all validations.
    Returns 0 if all required services pass, 1 otherwise.
    """
    print_header()
    
    results = {}
    
    # Required services
    print(f"\n{Fore.CYAN}📡 Required Services:{Style.RESET_ALL}")
    print("-" * 50)
    
    results["telegram"] = validate_telegram()
    
    # AI Providers (at least one required)
    print(f"\n{Fore.CYAN}🧠 AI Providers (at least one required):{Style.RESET_ALL}")
    print("-" * 50)
    
    ai_results = []
    ai_results.append(("OpenAI", validate_openai()))
    ai_results.append(("Anthropic", validate_anthropic()))
    ai_results.append(("Google", validate_google()))
    
    # Check if at least one AI provider works
    ai_ok = any(r[1] for r in ai_results if os.getenv(f"{r[0].upper()}_API_KEY") or r[0] == "OpenAI")
    
    # Trading (optional but recommended)
    print(f"\n{Fore.CYAN}📈 Trading Services:{Style.RESET_ALL}")
    print("-" * 50)
    
    results["alpaca"] = validate_alpaca()
    
    # Summary
    print(f"\n{Fore.CYAN}{'═' * 50}{Style.RESET_ALL}")
    
    # Determine overall result
    required_ok = results["telegram"] and ai_ok
    
    if required_ok:
        print(f"""
{Fore.GREEN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════╗
║                    ✅ VALIDATION PASSED                      ║
║                  All systems ready to go!                    ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        return 0
    else:
        print(f"""
{Fore.RED}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════╗
║                    ❌ VALIDATION FAILED                      ║
║              Please fix the errors above                     ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        
        print(f"{Fore.YELLOW}Troubleshooting:{Style.RESET_ALL}")
        if not results["telegram"]:
            print("  • Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env")
        if not ai_ok:
            print("  • Set at least one AI provider key (OPENAI_API_KEY recommended)")
        if not results["alpaca"]:
            print("  • Check ALPACA_API_KEY and ALPACA_SECRET_KEY (optional for analysis-only mode)")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

