"""
Script to verify .env configuration.
Checks if API keys are loaded correctly without revealing them.
"""
import os
from dotenv import load_dotenv

def check_env():
    """Checks for Alpaca credentials in .env file."""
    print("Checking environment configuration...\n")

    # Reload to ensure we get fresh values
    load_dotenv(override=True)

    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    paper_mode = os.getenv('ALPACA_PAPER')

    all_good = True

    # Check API Key
    if api_key:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        print(f"[OK] ALPACA_API_KEY found: {masked_key}")
    else:
        print("[FAIL] ALPACA_API_KEY is missing or empty.")
        all_good = False

    # Check Secret Key
    if secret_key:
        masked_secret = f"{secret_key[:4]}...{secret_key[-4:]}" if len(secret_key) > 8 else "****"
        print(f"[OK] ALPACA_SECRET_KEY found: {masked_secret}")
    else:
        print("[FAIL] ALPACA_SECRET_KEY is missing or empty.")
        all_good = False

    # Check Paper Mode
    if paper_mode:
        print(f"[OK] ALPACA_PAPER set to: {paper_mode}")
    else:
        print("[WARN] ALPACA_PAPER is missing. Defaulting to True (Safe Mode).")

    print("-" * 40)
    if all_good:
        print("SUCCESS: Environment is ready for trading.")
    else:
        print("ERROR: Please create a .env file with your credentials.")
        print("Example content:")
        print("ALPACA_API_KEY=PKxxxxxxxxxx")
        print("ALPACA_SECRET_KEY=skxxxxxxxxxx")
        print("ALPACA_PAPER=true")

if __name__ == "__main__":
    check_env()
