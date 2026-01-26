"""
Market Scanner - Pre-Market Gapper & Volume Scanner
Identifies "Stocks in Play" for automated watchlist generation.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("MarketScanner")

# ==================== SCANNER UNIVERSE ====================
# Top 200 most actively traded stocks to scan
# This serves as our scanning universe when we can't get a live screener

SCAN_UNIVERSE = [
    # Mega Caps
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH",
    "JPM", "V", "JNJ", "XOM", "WMT", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "WFC", "NKE", "VZ", "ADBE", "TXN", "CRM", "PM", "NEE",
    
    # High Beta / Momentum Names
    "AMD", "COIN", "MARA", "RIOT", "SOFI", "PLTR", "RIVN", "LCID", "NIO", "XPEV",
    "HOOD", "UPST", "AFRM", "RBLX", "U", "DKNG", "PENN", "CHWY", "PTON", "SNAP",
    "PINS", "TWLO", "ROKU", "ZM", "DOCU", "CRWD", "ZS", "NET", "DDOG", "MDB",
    "SNOW", "PATH", "ABNB", "DASH", "LYFT", "UBER", "SQ", "PYPL", "SHOP", "MELI",
    
    # Biotech / Healthcare
    "MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "ILMN", "ISRG", "DXCM", "ALGN",
    "SGEN", "BMRN", "EXAS", "HZNP", "UTHR", "NBIX", "SRPT", "ALNY", "RARE", "BLUE",
    
    # Energy
    "OXY", "SLB", "HAL", "DVN", "MRO", "APA", "FANG", "EOG", "PXD", "COP",
    
    # Financials
    "GS", "MS", "C", "BAC", "SCHW", "BX", "KKR", "APO", "TROW", "BLK",
    
    # Retail / Consumer
    "TGT", "LOW", "TJX", "ROST", "DG", "DLTR", "BBY", "ULTA", "LULU", "GPS",
    
    # Tech Growth
    "NOW", "PANW", "FTNT", "OKTA", "SPLK", "VEEV", "WDAY", "TEAM", "HUBS", "TTD",
    "BILL", "ZI", "CFLT", "ESTC", "GTLB", "IOT", "DOCN", "MNDY", "FROG", "BRZE",
    
    # Meme / Retail Favorites
    "GME", "AMC", "BB", "BBBY", "SPCE", "WISH", "CLOV", "WKHS", "GOEV", "NKLA",
    
    # Semis
    "INTC", "MU", "QCOM", "AMAT", "LRCX", "KLAC", "ASML", "TSM", "ON", "MCHP",
    
    # ETFs for Sector Gauges
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "ARKK",
    
    # China ADRs
    "BABA", "JD", "PDD", "BIDU", "LI", "BILI", "TME", "NTES", "TCEHY", "VNET",
    
    # Cannabis
    "TLRY", "CGC", "ACB", "SNDL", "CRON", "CURLF", "GTBIF", "TCNNF", "CRLBF", "TRUL",
]


class MarketScanner:
    """
    Scans for stocks in play based on:
    - Pre-market/Intraday Volume
    - Gap % (> 3% or < -3%)
    - Price filter (> $5)
    """
    
    def __init__(self):
        self.alpaca_client = None
        self.data_client = None
        self._init_alpaca()
    
    def _init_alpaca(self):
        """Initialize Alpaca clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            
            if api_key and secret_key:
                self.alpaca_client = TradingClient(api_key, secret_key, paper=True)
                self.data_client = StockHistoricalDataClient(api_key, secret_key)
                logger.info("Alpaca clients initialized for scanning")
            else:
                logger.warning("Alpaca credentials not found - using yfinance fallback")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
    
    def get_top_movers_alpaca(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top movers using Alpaca's snapshot API.
        Returns list of dicts with ticker info.
        """
        if not self.data_client:
            return []
        
        try:
            from alpaca.data.requests import StockSnapshotRequest
            
            # Get snapshots for our universe
            # Alpaca limits to 1000 symbols per request
            symbols = SCAN_UNIVERSE[:200]
            
            request = StockSnapshotRequest(symbol_or_symbols=symbols)
            snapshots = self.data_client.get_stock_snapshot(request)
            
            movers = []
            
            for symbol, snapshot in snapshots.items():
                try:
                    if not snapshot or not snapshot.daily_bar or not snapshot.prev_daily_bar:
                        continue
                    
                    current_price = snapshot.daily_bar.close
                    prev_close = snapshot.prev_daily_bar.close
                    volume = snapshot.daily_bar.volume
                    
                    # Skip if no valid data
                    if not current_price or not prev_close or prev_close == 0:
                        continue
                    
                    # Calculate gap %
                    gap_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    # Apply filters
                    if current_price < 5:  # Price filter
                        continue
                    if volume < 50000:  # Volume filter
                        continue
                    if abs(gap_pct) < 3:  # Gap filter
                        continue
                    
                    movers.append({
                        "ticker": symbol,
                        "price": round(current_price, 2),
                        "prev_close": round(prev_close, 2),
                        "gap_pct": round(gap_pct, 2),
                        "volume": volume,
                        "direction": "GAINER" if gap_pct > 0 else "LOSER"
                    })
                
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    continue
            
            # Sort by absolute gap %
            movers.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
            
            return movers[:limit]
        
        except Exception as e:
            logger.error(f"Alpaca snapshot scan failed: {e}")
            return []
    
    def get_top_movers_yfinance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fallback scanner using yfinance.
        Slower but works without Alpaca.
        """
        import yfinance as yf
        
        movers = []
        
        # Scan in batches for efficiency
        batch_size = 50
        symbols_to_scan = SCAN_UNIVERSE[:100]  # Limit for speed
        
        for i in range(0, len(symbols_to_scan), batch_size):
            batch = symbols_to_scan[i:i + batch_size]
            
            try:
                # Download batch data
                tickers = yf.Tickers(" ".join(batch))
                
                for symbol in batch:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if not ticker:
                            continue
                        
                        info = ticker.info
                        
                        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
                        volume = info.get("volume") or info.get("regularMarketVolume", 0)
                        
                        if not current_price or not prev_close or prev_close == 0:
                            continue
                        
                        gap_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        # Apply filters
                        if current_price < 5:
                            continue
                        if volume < 50000:
                            continue
                        if abs(gap_pct) < 3:
                            continue
                        
                        movers.append({
                            "ticker": symbol,
                            "price": round(current_price, 2),
                            "prev_close": round(prev_close, 2),
                            "gap_pct": round(gap_pct, 2),
                            "volume": volume,
                            "direction": "GAINER" if gap_pct > 0 else "LOSER"
                        })
                    
                    except Exception as e:
                        logger.debug(f"YF error for {symbol}: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"YFinance batch error: {e}")
                continue
        
        # Sort by absolute gap %
        movers.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
        
        return movers[:limit]
    
    def get_top_gainers_and_losers(self, limit: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Main scanner function using multi-source data.
        Returns (list of tickers, list of full mover data).
        
        Priority:
        1. FMP (Financial Modeling Prep) - Best for movers/screeners
        2. Alpaca
        3. yfinance (fallback)
        
        Criteria:
        - Price > $5
        - Volume > 500,000
        - Gap > 3% or < -3%
        """
        logger.info(f"Starting market scan for top {limit} movers...")
        
        # Try FMP first (best source for movers)
        try:
            from src.market_data import get_market_movers, FMP_API_KEY
            
            if FMP_API_KEY:
                logger.info("Using FMP for market movers (PRO mode)")
                tickers, fmp_movers = get_market_movers(limit)
                
                if fmp_movers:
                    # Convert FMP format to our format
                    movers = []
                    for m in fmp_movers:
                        direction = "GAINER" if m.get('change_pct', 0) > 0 else "LOSER"
                        movers.append({
                            "ticker": m.get('symbol'),
                            "price": m.get('price', 0),
                            "gap_pct": m.get('change_pct', 0),
                            "volume": m.get('volume', 0),
                            "direction": direction,
                            "source": "FMP"
                        })
                    
                    gainers = [m for m in movers if m["direction"] == "GAINER"]
                    losers = [m for m in movers if m["direction"] == "LOSER"]
                    
                    logger.info(f"FMP scan complete: {len(gainers)} gainers, {len(losers)} losers")
                    return tickers, movers
        except ImportError:
            logger.debug("market_data module not available, using fallback")
        except Exception as e:
            logger.warning(f"FMP scan failed: {e}")
        
        # Try Alpaca (faster, real-time)
        movers = self.get_top_movers_alpaca(limit * 2)
        
        # Fallback to yfinance
        if not movers:
            logger.info("Using yfinance fallback for scanning")
            movers = self.get_top_movers_yfinance(limit * 2)
        
        if not movers:
            logger.warning("No movers found in scan")
            return [], []
        
        # Split into gainers and losers
        gainers = [m for m in movers if m["direction"] == "GAINER"][:limit]
        losers = [m for m in movers if m["direction"] == "LOSER"][:limit]
        
        # Combine (prioritize gainers for long bias)
        combined = gainers + losers
        combined = combined[:limit]
        
        tickers = [m["ticker"] for m in combined]
        
        logger.info(f"Scan complete: Found {len(gainers)} gainers, {len(losers)} losers")
        
        return tickers, combined
    
    def get_premarket_movers(self, limit: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Specifically scan for pre-market activity.
        Uses extended hours data if available.
        """
        # For now, same as regular scan
        # Could be enhanced with pre-market specific endpoints
        return self.get_top_gainers_and_losers(limit)
    
    def get_unusual_volume(self, min_volume_ratio: float = 2.0, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find stocks with unusual volume (volume > X times average).
        """
        import yfinance as yf
        
        unusual = []
        
        for symbol in SCAN_UNIVERSE[:50]:  # Quick scan
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                volume = info.get("volume", 0)
                avg_volume = info.get("averageVolume", 1)
                price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                
                if avg_volume == 0 or price < 5:
                    continue
                
                vol_ratio = volume / avg_volume
                
                if vol_ratio >= min_volume_ratio:
                    unusual.append({
                        "ticker": symbol,
                        "price": round(price, 2),
                        "volume": volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": round(vol_ratio, 2)
                    })
            
            except Exception as e:
                continue
        
        unusual.sort(key=lambda x: x["volume_ratio"], reverse=True)
        return unusual[:limit]
    
    def format_scan_results(self, movers: List[Dict[str, Any]]) -> str:
        """Format scan results for Telegram message."""
        if not movers:
            return "❌ לא נמצאו מניות בתנועה משמעותית"
        
        lines = ["🔎 **תוצאות סריקה - Stocks in Play:**\n"]
        
        for i, m in enumerate(movers, 1):
            direction_emoji = "🟢" if m["direction"] == "GAINER" else "🔴"
            gap_sign = "+" if m["gap_pct"] > 0 else ""
            
            lines.append(
                f"{i}. {direction_emoji} **{m['ticker']}** @ ${m['price']:.2f}\n"
                f"   Gap: {gap_sign}{m['gap_pct']:.1f}% | Vol: {m['volume']:,}"
            )
        
        return "\n".join(lines)


# ==================== MODULE-LEVEL FUNCTIONS ====================

_scanner_instance = None

def get_scanner() -> MarketScanner:
    """Get singleton scanner instance."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = MarketScanner()
    return _scanner_instance


def get_top_gainers_and_losers(limit: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Convenience function for scanning."""
    scanner = get_scanner()
    return scanner.get_top_gainers_and_losers(limit)


if __name__ == "__main__":
    # Test the scanner
    logging.basicConfig(level=logging.INFO)
    
    scanner = get_scanner()
    tickers, movers = scanner.get_top_gainers_and_losers(10)
    
    print("\n" + "="*50)
    print("TOP MOVERS SCAN RESULTS")
    print("="*50)
    
    for m in movers:
        print(f"{m['direction']:6} | {m['ticker']:5} | ${m['price']:>8.2f} | Gap: {m['gap_pct']:+6.2f}% | Vol: {m['volume']:>12,}")
    
    print("\nTickers for watchlist:", tickers)

