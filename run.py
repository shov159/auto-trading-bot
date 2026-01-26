#!/usr/bin/env python3
"""
CIO Trading Bot - Master Launcher
==================================
Runs pre-flight checks and launches the Telegram bot.
"""
import os
import sys
import io

# =============================================================================
# WINDOWS UTF-8 SETUP
# =============================================================================
if sys.platform == 'win32':
    # Enable UTF-8 output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Also try to set console mode
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass

# =============================================================================
# COLOR SETUP (works even before colorama is imported)
# =============================================================================
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        CYAN = GREEN = RED = YELLOW = WHITE = MAGENTA = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

# =============================================================================
# BANNER
# =============================================================================
BANNER = f"""
{Fore.CYAN}{Style.BRIGHT}
+======================================================================+
|                                                                      |
|       CIO TRADE - Chief Investment Officer Trading Bot               |
|                                                                      |
|    ######  ##  #######       ######## ########     ###    ########   |
|   ##    ## ## ##     ##         ##    ##     ##   ## ##   ##     ##  |
|   ##       ## ##     ##         ##    ##     ##  ##   ##  ##     ##  |
|   ##       ## ##     ##         ##    ########  ##     ## ##     ##  |
|   ##       ## ##     ##         ##    ##   ##   ######### ##     ##  |
|   ##    ## ## ##     ##         ##    ##    ##  ##     ## ##     ##  |
|    ######  ##  #######          ##    ##     ## ##     ## ########   |
|                                                                      |
+======================================================================+
{Style.RESET_ALL}"""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
def preflight_check() -> bool:
    """
    Verify all required environment variables are set.
    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{Fore.CYAN}🔍 Running Pre-Flight Checks...{Style.RESET_ALL}\n")
    
    required_vars = {
        "TELEGRAM_TOKEN": "Telegram Bot Token",
        "TELEGRAM_CHAT_ID": "Telegram Chat ID",
    }
    
    # At least one AI provider required
    ai_providers = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google Gemini"
    }
    
    # Alpaca (optional but recommended)
    alpaca_vars = {
        "ALPACA_API_KEY": "Alpaca API Key",
        "ALPACA_SECRET_KEY": "Alpaca Secret Key"
    }
    
    all_pass = True
    
    # Check required vars
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  {Fore.GREEN}✅ {description}: Set{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}❌ {description}: MISSING ({var}){Style.RESET_ALL}")
            all_pass = False
    
    # Check AI providers (at least one needed)
    ai_found = False
    ai_provider_name = None
    print(f"\n  {Fore.CYAN}AI Providers:{Style.RESET_ALL}")
    for var, name in ai_providers.items():
        value = os.getenv(var)
        if value:
            print(f"    {Fore.GREEN}✅ {name}: Set{Style.RESET_ALL}")
            if not ai_found:
                ai_found = True
                ai_provider_name = name
        else:
            print(f"    {Fore.YELLOW}⚪ {name}: Not set{Style.RESET_ALL}")
    
    if not ai_found:
        print(f"\n  {Fore.RED}❌ ERROR: At least one AI provider API key is required!{Style.RESET_ALL}")
        all_pass = False
    else:
        print(f"\n  {Fore.GREEN}✅ Using: {ai_provider_name}{Style.RESET_ALL}")
    
    # Check Alpaca (optional)
    print(f"\n  {Fore.CYAN}Trading (Alpaca):{Style.RESET_ALL}")
    alpaca_ok = True
    for var, description in alpaca_vars.items():
        value = os.getenv(var)
        if value:
            print(f"    {Fore.GREEN}✅ {description}: Set{Style.RESET_ALL}")
        else:
            print(f"    {Fore.YELLOW}⚠️ {description}: Not set (execution disabled){Style.RESET_ALL}")
            alpaca_ok = False
    
    if not alpaca_ok:
        print(f"\n  {Fore.YELLOW}⚠️ Alpaca not configured - One-Click Execution will be disabled{Style.RESET_ALL}")
    
    # Data Sources & FMP
    print(f"\n  {Fore.CYAN}Data Sources:{Style.RESET_ALL}")
    fmp_key = os.getenv("FMP_API_KEY", "")
    if fmp_key:
        print(f"    {Fore.GREEN}✅ FMP API Key: Set (PRO movers + news){Style.RESET_ALL}")
    else:
        print(f"    {Fore.YELLOW}⚪ FMP API Key: Not set{Style.RESET_ALL}")
        print(f"       {Fore.WHITE}Get free key: https://financialmodelingprep.com{Style.RESET_ALL}")
    
    print(f"    {Fore.GREEN}✅ Alpaca Data: Available{Style.RESET_ALL}" if alpaca_ok else 
          f"    {Fore.YELLOW}⚪ Alpaca Data: Limited{Style.RESET_ALL}")
    print(f"    {Fore.GREEN}✅ yfinance: Fallback ready{Style.RESET_ALL}")
    
    # Debug mode check
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    print(f"\n  {Fore.CYAN}Debug Mode: {Fore.MAGENTA if debug_mode else Fore.WHITE}{'ON' if debug_mode else 'OFF'}{Style.RESET_ALL}")
    
    return all_pass


def print_startup_info():
    """Print startup configuration summary."""
    risk = os.getenv("RISK_PER_TRADE", "50")
    risk_pct = os.getenv("RISK_PERCENTAGE", "0.005")
    max_risk = os.getenv("MAX_RISK_AMOUNT", "2000")
    use_dynamic = os.getenv("USE_DYNAMIC_RISK", "true").lower() == "true"
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    turbo = os.getenv("TURBO_MODE", "false").lower() == "true"
    auto_trade = os.getenv("AUTO_TRADE", "false").lower() == "true"
    auto_min = os.getenv("AUTO_TRADE_MIN_CONVICTION", "HIGH")
    
    mode_text = "PAPER TRADING" if paper else "!! LIVE TRADING !!"
    mode_color = Fore.GREEN if paper else Fore.RED
    
    # Risk display
    if use_dynamic:
        risk_mode = f"{Fore.CYAN}DYNAMIC{Style.RESET_ALL}"
        risk_info = f"{float(risk_pct)*100:.2f}% of equity (max ${max_risk})"
    else:
        risk_mode = f"{Fore.YELLOW}FIXED{Style.RESET_ALL}"
        risk_info = f"${risk}"
    
    # Turbo/Auto-trade display
    scan_interval = "5 min" if turbo else "30 min"
    turbo_display = f"{Fore.YELLOW}⚡ ON{Style.RESET_ALL}" if turbo else f"{Fore.WHITE}OFF{Style.RESET_ALL}"
    auto_display = f"{Fore.RED}🤖 ON ({auto_min}){Style.RESET_ALL}" if auto_trade else f"{Fore.WHITE}OFF{Style.RESET_ALL}"
    
    print(f"""
{Fore.CYAN}+----------------------------------------------------------------------+
|  CONFIGURATION SUMMARY                                               |
+----------------------------------------------------------------------+{Style.RESET_ALL}
  [*] Mode: {mode_color}{mode_text}{Style.RESET_ALL}
  [$] Risk Mode: {risk_mode} - {risk_info}
  [⚡] Turbo Mode: {turbo_display} (scan every {scan_interval})
  [🤖] Auto-Trade: {auto_display}
  [T] Pre-market scan: 16:00 IL / 9:00 AM NY
    
{Fore.GREEN}======================================================================={Style.RESET_ALL}
""")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Print banner
    print(BANNER)
    
    # Run pre-flight checks
    if not preflight_check():
        print(f"\n{Fore.RED}{Style.BRIGHT}❌ PRE-FLIGHT CHECK FAILED{Style.RESET_ALL}")
        print(f"{Fore.RED}Please set the required environment variables in .env file{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Example .env file:{Style.RESET_ALL}")
        print("""
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
OPENAI_API_KEY=sk-your-key-here
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true
DEBUG_MODE=false
RISK_PER_TRADE=50
""")
        sys.exit(1)
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}✅ ALL CHECKS PASSED{Style.RESET_ALL}")
    print_startup_info()
    
    # Clear stale cache on startup with STRICT 15-min TTL
    print(f"\n{Fore.CYAN}🧹 Clearing stale cache...{Style.RESET_ALL}")
    try:
        from src.analysis_cache import get_analysis_cache
        
        # Determine TTL based on turbo mode
        turbo = os.getenv("TURBO_MODE", "false").lower() == "true"
        cache_ttl = 10.0 if turbo else 15.0  # STRICT: 15 min normal, 10 min turbo
        
        # Create fresh cache with proper TTL
        cache = get_analysis_cache(max_age_minutes=cache_ttl, force_new=True)
        
        # Clear ALL old data on startup (prevents trading on stale analysis)
        cleared = cache.clear_on_startup()
        if cleared > 0:
            print(f"  {Fore.YELLOW}⚠️ Cleared {cleared} old cache entries (fresh start){Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}✅ Cache is empty (fresh start){Style.RESET_ALL}")
        
        print(f"  {Fore.WHITE}📊 Cache TTL: {cache_ttl} minutes{Style.RESET_ALL}")
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠️ Could not clean cache: {e}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}🚀 Starting CIO Trading Bot...{Style.RESET_ALL}\n")
    
    try:
        # Import and run the bot
        from src.telegram_bot import main as bot_main
        bot_main()
    
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️ Keyboard interrupt received{Style.RESET_ALL}")
        print(f"{Fore.CYAN}👋 Shutting down gracefully...{Style.RESET_ALL}")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}💥 FATAL ERROR:{Style.RESET_ALL}")
        print(f"{Fore.RED}{e}{Style.RESET_ALL}")
        
        # Show full traceback in debug mode
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            import traceback
            print(f"\n{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()

