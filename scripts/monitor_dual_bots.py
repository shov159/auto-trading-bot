import time
import os
import signal
import sys
from collections import deque

def tail_files(files):
    """
    Tails multiple files simultaneously.
    """
    files_handles = {}
    for name, path in files.items():
        if os.path.exists(path):
            f = open(path, 'r', encoding='utf-8')
            # Go to the end of the file
            f.seek(0, 2)
            files_handles[name] = f
        else:
            print(f"Waiting for {path} to be created...")

    print("\n--- Monitoring Dual Bots (SPY + Crypto) ---")
    print("Press Ctrl+C to stop.\n")

    while True:
        data_found = False
        for name, f in files_handles.items():
            line = f.readline()
            if line:
                data_found = True
                line = line.strip()
                
                # Color/Highlighting logic (Simple text markers for now)
                prefix = f"[{name}]"
                
                if "ERROR" in line:
                    print(f"🔴 {prefix} {line}")
                elif "TRADE EXECUTION" in line or "AI Signal" in line:
                    print(f"🟢 {prefix} {line}")
                elif "Max Drawdown" in line:
                    print(f"⚠️ {prefix} {line}")
                else:
                    print(f"{prefix} {line}")
        
        # Check for new files if not all opened
        if len(files_handles) < len(files):
             for name, path in files.items():
                if name not in files_handles and os.path.exists(path):
                    f = open(path, 'r', encoding='utf-8')
                    f.seek(0, 2)
                    files_handles[name] = f
                    print(f"[{name}] Log file detected and opened.")

        if not data_found:
            time.sleep(0.5)

if __name__ == "__main__":
    log_files = {
        "SPY": "logs/spy_bot.log",
        "CRYPTO": "logs/crypto_bot.log"
    }
    
    try:
        tail_files(log_files)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

