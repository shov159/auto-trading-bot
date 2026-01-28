"""
Market Data - Multi-Source Data Aggregator
==========================================
Fetches market data from multiple sources with fallback logic:
1. Alpaca (most accurate for quotes)
2. FMP (Financial Modeling Prep - best for movers/screeners)
3. yfinance (free fallback)
"""
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()

from src.logger import log_info, log_warn, log_error, log_ok, log_debug

# =============================================================================
# API KEYS
# =============================================================================
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# =============================================================================
# FMP DATA FUNCTIONS
# =============================================================================

def get_fmp_movers(mover_type: str = "gainers", limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch market movers from FMP API.

    Args:
        mover_type: 'gainers', 'losers', or 'actives'
        limit: Max number of results

    Returns:
        List of mover dicts with symbol, price, change, volume
    """
    if not FMP_API_KEY:
        log_warn("FMP_API_KEY not set - cannot fetch movers")
        return []

    try:
        endpoint_map = {
            "gainers": "stock_market/gainers",
            "losers": "stock_market/losers",
            "actives": "stock_market/actives"
        }

        endpoint = endpoint_map.get(mover_type, "stock_market/gainers")
        url = f"{FMP_BASE_URL}/{endpoint}"
        params = {"apikey": FMP_API_KEY}

        log_debug(f"Fetching FMP {mover_type}...")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Filter: Price > $5, Volume > 500K
            filtered = []
            for item in data[:limit * 2]:  # Get extra to filter
                price = item.get("price", 0)
                volume = item.get("volume", 0)
                change_pct = item.get("changesPercentage", 0)

                # Apply filters
                if price >= 5 and volume >= 500000:
                    filtered.append({
                        "symbol": item.get("symbol", ""),
                        "name": item.get("name", ""),
                        "price": price,
                        "change": item.get("change", 0),
                        "change_pct": change_pct,
                        "volume": volume,
                        "source": "FMP"
                    })

                    if len(filtered) >= limit:
                        break

            log_ok(f"FMP {mover_type}: Found {len(filtered)} stocks")
            return filtered
        else:
            log_error(f"FMP API error: {response.status_code}")
            return []

    except Exception as e:
        log_error(f"FMP movers fetch error: {e}")
        return []


def get_fmp_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time quote from FMP.

    Args:
        ticker: Stock symbol

    Returns:
        Quote dict with price, volume, change, etc.
    """
    if not FMP_API_KEY:
        return None

    try:
        url = f"{FMP_BASE_URL}/quote/{ticker.upper()}"
        params = {"apikey": FMP_API_KEY}

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                quote = data[0]
                return {
                    "symbol": quote.get("symbol"),
                    "price": quote.get("price", 0),
                    "change": quote.get("change", 0),
                    "change_pct": quote.get("changesPercentage", 0),
                    "volume": quote.get("volume", 0),
                    "avg_volume": quote.get("avgVolume", 0),
                    "high": quote.get("dayHigh", 0),
                    "low": quote.get("dayLow", 0),
                    "open": quote.get("open", 0),
                    "previous_close": quote.get("previousClose", 0),
                    "market_cap": quote.get("marketCap", 0),
                    "pe": quote.get("pe", 0),
                    "eps": quote.get("eps", 0),
                    "year_high": quote.get("yearHigh", 0),
                    "year_low": quote.get("yearLow", 0),
                    "source": "FMP"
                }
        return None

    except Exception as e:
        log_debug(f"FMP quote error for {ticker}: {e}")
        return None


def get_fmp_stock_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch news for a specific stock from FMP."""
    if not FMP_API_KEY:
        return []

    try:
        url = f"{FMP_BASE_URL}/stock_news"
        params = {
            "tickers": ticker.upper(),
            "limit": limit,
            "apikey": FMP_API_KEY
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        return []

    except Exception as e:
        log_debug(f"FMP news error: {e}")
        return []


# =============================================================================
# ALPACA DATA FUNCTIONS
# =============================================================================

def get_alpaca_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time quote from Alpaca.
    Most accurate for stocks during market hours.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        # Get snapshot for comprehensive data
        request = StockSnapshotRequest(symbol_or_symbols=ticker.upper())
        snapshot = client.get_stock_snapshot(request)

        if ticker.upper() in snapshot:
            snap = snapshot[ticker.upper()]

            latest_trade = snap.latest_trade
            daily_bar = snap.daily_bar

            return {
                "symbol": ticker.upper(),
                "price": float(latest_trade.price) if latest_trade else 0,
                "volume": int(daily_bar.volume) if daily_bar else 0,
                "high": float(daily_bar.high) if daily_bar else 0,
                "low": float(daily_bar.low) if daily_bar else 0,
                "open": float(daily_bar.open) if daily_bar else 0,
                "vwap": float(daily_bar.vwap) if daily_bar else 0,
                "source": "Alpaca"
            }
        return None

    except Exception as e:
        log_debug(f"Alpaca quote error for {ticker}: {e}")
        return None


def get_alpaca_movers() -> List[Dict[str, Any]]:
    """
    Get market movers from Alpaca.
    Alpaca doesn't have a direct movers endpoint, so this scans a universe.
    """
    # Alpaca doesn't have a movers endpoint - return empty
    # We use FMP for movers instead
    return []


# =============================================================================
# YFINANCE FALLBACK
# =============================================================================

def get_yfinance_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """Fallback quote using yfinance."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
            return None

        price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        return {
            "symbol": ticker.upper(),
            "price": price,
            "change": info.get('regularMarketChange', 0),
            "change_pct": info.get('regularMarketChangePercent', 0),
            "volume": info.get('volume', 0),
            "avg_volume": info.get('averageVolume', 0),
            "high": info.get('dayHigh', 0),
            "low": info.get('dayLow', 0),
            "open": info.get('open', 0),
            "previous_close": info.get('previousClose', 0),
            "market_cap": info.get('marketCap', 0),
            "year_high": info.get('fiftyTwoWeekHigh', 0),
            "year_low": info.get('fiftyTwoWeekLow', 0),
            "source": "yfinance"
        }

    except Exception as e:
        log_debug(f"yfinance quote error for {ticker}: {e}")
        return None


def get_yfinance_movers() -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Fallback movers using yfinance.
    Scans a predefined universe since yfinance doesn't have screener.
    """
    try:
        import yfinance as yf

        # Scan popular stocks
        universe = [
            "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL",
            "COIN", "MARA", "RIOT", "PLTR", "SOFI", "NIO", "RIVN", "LCID",
            "GME", "AMC", "SPY", "QQQ", "ARKK", "SQ", "PYPL", "SHOP",
            'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'XOM', 'UNH', 'V', 'PG', 'MA', 'JNJ', 'HD', 'MRK', 'COST',
    'ABBV', 'CVX', 'CRM', 'BAC', 'WMT', 'NFLX', 'PEP', 'KO', 'TMO',
    'DIS', 'ADBE', 'CSCO', 'ACN', 'MCD', 'INTC', 'CMCSA', 'PFE', 'NKE', 'VZ',
    'INTU', 'AMGN', 'TXN', 'DHR', 'UNP', 'PM', 'SPGI', 'CAT', 'HON', 'COP',
    'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE'
        ]

        movers = []

        for ticker in universe:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")

                if len(hist) >= 2:
                    today = hist.iloc[-1]
                    yesterday = hist.iloc[-2]

                    price = today['Close']
                    change = price - yesterday['Close']
                    change_pct = (change / yesterday['Close']) * 100
                    volume = today['Volume']

                    if abs(change_pct) > 2 and volume > 500000 and price > 5:
                        movers.append({
                            "symbol": ticker,
                            "price": price,
                            "change": change,
                            "change_pct": change_pct,
                            "volume": volume,
                            "source": "yfinance"
                        })
            except:
                continue

        # Sort by change percentage
        movers.sort(key=lambda x: abs(x.get('change_pct', 0)), reverse=True)

        tickers = [m['symbol'] for m in movers[:10]]
        return tickers, movers[:10]

    except Exception as e:
        log_error(f"yfinance movers error: {e}")
        return [], []


# =============================================================================
# UNIFIED API FUNCTIONS (Multi-Source with Fallback)
# =============================================================================

def get_current_price(ticker: str) -> Optional[float]:
    """
    Get current price with multi-source fallback.

    Priority:
    1. Alpaca (most accurate during market hours)
    2. FMP
    3. yfinance
    """
    # Try Alpaca first
    quote = get_alpaca_quote(ticker)
    if quote and quote.get('price', 0) > 0:
        log_debug(f"Price for {ticker} from Alpaca: ${quote['price']}")
        return quote['price']

    # Try FMP
    quote = get_fmp_quote(ticker)
    if quote and quote.get('price', 0) > 0:
        log_debug(f"Price for {ticker} from FMP: ${quote['price']}")
        return quote['price']

    # Fallback to yfinance
    quote = get_yfinance_quote(ticker)
    if quote and quote.get('price', 0) > 0:
        log_debug(f"Price for {ticker} from yfinance: ${quote['price']}")
        return quote['price']

    log_warn(f"Could not get price for {ticker} from any source")
    return None


def get_full_quote(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive quote data with multi-source fallback.
    """
    # Try FMP first (has most data)
    quote = get_fmp_quote(ticker)
    if quote:
        return quote

    # Try Alpaca
    quote = get_alpaca_quote(ticker)
    if quote:
        return quote

    # Fallback to yfinance
    quote = get_yfinance_quote(ticker)
    if quote:
        return quote

    return {"symbol": ticker, "price": 0, "source": "none"}


def get_market_movers(limit: int = 10) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Get market movers with FMP priority.

    Returns:
        (tickers_list, full_data_list)
    """
    # Try FMP first (best source for movers)
    if FMP_API_KEY:
        log_info("Fetching movers from FMP...")

        gainers = get_fmp_movers("gainers", limit=limit // 2 + 2)
        losers = get_fmp_movers("losers", limit=limit // 2 + 2)

        # Combine and sort by absolute change
        all_movers = gainers + losers
        all_movers.sort(key=lambda x: abs(x.get('change_pct', 0)), reverse=True)

        if all_movers:
            tickers = [m['symbol'] for m in all_movers[:limit]]
            log_ok(f"FMP movers: {', '.join(tickers[:5])}...")
            return tickers, all_movers[:limit]

    # Fallback to yfinance
    log_warn("FMP unavailable, falling back to yfinance for movers")
    return get_yfinance_movers()


def check_data_sources() -> Dict[str, bool]:
    """Check which data sources are available."""
    sources = {
        "alpaca": bool(ALPACA_API_KEY and ALPACA_SECRET_KEY),
        "fmp": bool(FMP_API_KEY),
        "yfinance": True  # Always available
    }

    # Test FMP connection
    if sources["fmp"]:
        try:
            test = get_fmp_quote("AAPL")
            sources["fmp_working"] = test is not None
        except:
            sources["fmp_working"] = False

    return sources


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("\n=== Data Source Check ===")
    sources = check_data_sources()
    for src, status in sources.items():
        print(f"  {src}: {'✅' if status else '❌'}")

    print("\n=== Testing Price Fetch ===")
    price = get_current_price("NVDA")
    print(f"NVDA Price: ${price}")

    print("\n=== Testing Market Movers ===")
    tickers, movers = get_market_movers(5)
    print(f"Movers: {tickers}")
    for m in movers:
        print(f"  {m['symbol']}: ${m['price']:.2f} ({m['change_pct']:+.2f}%) Vol: {m['volume']:,}")
