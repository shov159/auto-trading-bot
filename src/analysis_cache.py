"""
Analysis Cache - Smart Caching for AI Analysis Results
=======================================================
Reduces API calls by caching analysis results and only
re-analyzing when price moves significantly or cache expires.
"""
import json
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from src.logger import log_debug, log_info, log_ok, log_warn


@dataclass
class CacheEntry:
    """Cached analysis entry."""
    ticker: str
    price_at_analysis: float
    analysis_result: Dict[str, Any]
    timestamp: str  # ISO format
    hit_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        return cls(**data)


class AnalysisCache:
    """
    Smart cache for AI analysis results.
    
    Caching Strategy:
    - Cache key: TICKER
    - Cache valid if:
      1. Entry age < max_age_minutes (default 15 - STRICT!)
      2. Price change < price_threshold_pct (default 0.5%)
    - If either condition fails, cache is DELETED (not just invalidated)
    
    TURBO MODE: Uses 10 min TTL
    NORMAL MODE: Uses 15 min TTL (was 60 - too stale!)
    """
    
    def __init__(
        self,
        max_age_minutes: float = 15.0,  # STRICT: 15 min default (was 60)
        price_threshold_pct: float = 0.5,
        cache_file: Optional[str] = None
    ):
        """
        Args:
            max_age_minutes: Maximum age of cache entry in minutes
            price_threshold_pct: Price change threshold to invalidate cache (0.5 = 0.5%)
            cache_file: Optional path to persist cache to disk
        """
        self.max_age = timedelta(minutes=max_age_minutes)
        self.price_threshold = price_threshold_pct / 100.0  # Convert to decimal
        self.cache_file = cache_file
        
        self._lock = threading.Lock()
        self._cache: Dict[str, CacheEntry] = {}
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        
        # Load from disk if file specified
        if cache_file:
            self._load_from_disk()
        
        log_info(f"AnalysisCache initialized: max_age={max_age_minutes}min, threshold={price_threshold_pct}%")
    
    def _load_from_disk(self):
        """Load cache from disk file."""
        try:
            if self.cache_file and Path(self.cache_file).exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for ticker, entry_data in data.get('entries', {}).items():
                        self._cache[ticker] = CacheEntry.from_dict(entry_data)
                    log_info(f"Loaded {len(self._cache)} cached entries from disk")
        except Exception as e:
            log_warn(f"Failed to load cache from disk: {e}")
    
    def _save_to_disk(self):
        """Persist cache to disk."""
        try:
            if self.cache_file:
                data = {
                    'entries': {k: v.to_dict() for k, v in self._cache.items()},
                    'stats': {
                        'hits': self._hits,
                        'misses': self._misses,
                        'invalidations': self._invalidations
                    },
                    'saved_at': datetime.now().isoformat()
                }
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_warn(f"Failed to save cache to disk: {e}")
    
    def _is_entry_valid(self, entry: CacheEntry, current_price: float) -> Tuple[bool, str]:
        """
        Check if a cache entry is still valid.
        
        Returns:
            (is_valid, reason) - tuple of validity and explanation
        """
        # Check age
        try:
            entry_time = datetime.fromisoformat(entry.timestamp)
            age = datetime.now() - entry_time
            
            if age > self.max_age:
                return False, f"expired ({age.total_seconds()/60:.0f}min old)"
        except (ValueError, TypeError):
            return False, "invalid timestamp"
        
        # Check price change
        if entry.price_at_analysis > 0 and current_price > 0:
            price_change = abs(current_price - entry.price_at_analysis) / entry.price_at_analysis
            
            if price_change > self.price_threshold:
                pct = price_change * 100
                return False, f"price moved {pct:.2f}% (threshold: {self.price_threshold*100:.1f}%)"
        
        return True, "valid"
    
    def get(self, ticker: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis if valid.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current market price
            
        Returns:
            Cached analysis result if valid, None otherwise
        """
        with self._lock:
            entry = self._cache.get(ticker.upper())
            
            if not entry:
                self._misses += 1
                log_debug(f"Cache MISS for {ticker}: no entry")
                return None
            
            is_valid, reason = self._is_entry_valid(entry, current_price)
            
            if is_valid:
                # Cache hit!
                self._hits += 1
                entry.hit_count += 1
                
                # Add cache metadata to result
                result = entry.analysis_result.copy()
                result['_cached'] = True
                result['_cache_age_minutes'] = (datetime.now() - datetime.fromisoformat(entry.timestamp)).total_seconds() / 60
                result['_cache_price'] = entry.price_at_analysis
                result['_cache_hits'] = entry.hit_count
                
                log_ok(f"Cache HIT for {ticker}: {reason} (hits: {entry.hit_count})")
                return result
            else:
                # Invalid - remove from cache
                self._invalidations += 1
                del self._cache[ticker.upper()]
                log_debug(f"Cache INVALIDATED for {ticker}: {reason}")
                return None
    
    def set(self, ticker: str, price: float, analysis: Dict[str, Any]):
        """
        Store analysis result in cache.
        
        Args:
            ticker: Stock ticker symbol
            price: Price at time of analysis
            analysis: Analysis result to cache
        """
        with self._lock:
            # Remove cache metadata if present (don't cache the cache info)
            clean_analysis = {k: v for k, v in analysis.items() if not k.startswith('_cache')}
            
            entry = CacheEntry(
                ticker=ticker.upper(),
                price_at_analysis=price,
                analysis_result=clean_analysis,
                timestamp=datetime.now().isoformat(),
                hit_count=0
            )
            
            self._cache[ticker.upper()] = entry
            log_debug(f"Cache SET for {ticker} @ ${price:.2f}")
            
            # Persist to disk periodically (every 5 entries)
            if len(self._cache) % 5 == 0:
                self._save_to_disk()
    
    def invalidate(self, ticker: str):
        """Manually invalidate a cache entry."""
        with self._lock:
            if ticker.upper() in self._cache:
                del self._cache[ticker.upper()]
                self._invalidations += 1
                log_debug(f"Cache manually invalidated for {ticker}")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            log_info(f"Cache cleared ({count} entries removed)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            
            return {
                'entries': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'invalidations': self._invalidations,
                'hit_rate_pct': round(hit_rate, 1),
                'total_requests': total
            }
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for ticker, entry in self._cache.items():
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    if now - entry_time > self.max_age:
                        expired.append(ticker)
                except (ValueError, TypeError):
                    expired.append(ticker)
            
            for ticker in expired:
                del self._cache[ticker]
            
            if expired:
                log_debug(f"Cleaned up {len(expired)} expired cache entries")
            
            return len(expired)
    
    def auto_expire(self) -> int:
        """
        STRICT: Delete ALL entries older than max_age immediately.
        Call this on startup and periodically to ensure no stale data.
        
        Returns:
            Number of entries deleted
        """
        with self._lock:
            now = datetime.now()
            to_delete = []
            
            for ticker, entry in self._cache.items():
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    age = now - entry_time
                    
                    if age > self.max_age:
                        age_mins = age.total_seconds() / 60
                        log_warn(f"Auto-expiring stale cache: {ticker} ({age_mins:.0f} min old)")
                        to_delete.append(ticker)
                except (ValueError, TypeError):
                    to_delete.append(ticker)
            
            for ticker in to_delete:
                del self._cache[ticker]
                self._invalidations += 1
            
            if to_delete:
                log_info(f"Auto-expired {len(to_delete)} stale cache entries")
                self._save_to_disk()
            
            return len(to_delete)
    
    def clear_on_startup(self):
        """
        Clear cache on bot startup to ensure fresh data.
        Prevents trading on yesterday's stale analysis.
        """
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        
        # Clear disk cache too
        if self.cache_file:
            try:
                Path(self.cache_file).unlink(missing_ok=True)
            except:
                pass
        
        log_info(f"Cache cleared on startup ({count} entries removed)")
        return count
    
    def get_cached_tickers(self) -> Dict[str, Dict]:
        """Get summary of all cached tickers."""
        with self._lock:
            result = {}
            now = datetime.now()
            
            for ticker, entry in self._cache.items():
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    age_mins = (now - entry_time).total_seconds() / 60
                    
                    result[ticker] = {
                        'price': entry.price_at_analysis,
                        'age_minutes': round(age_mins, 1),
                        'action': entry.analysis_result.get('action', 'UNKNOWN'),
                        'conviction': entry.analysis_result.get('conviction', 'UNKNOWN'),
                        'hit_count': entry.hit_count
                    }
                except (ValueError, TypeError):
                    pass
            
            return result


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_cache_instance: Optional[AnalysisCache] = None


def get_analysis_cache(
    max_age_minutes: float = 15.0,  # STRICT default: 15 min
    price_threshold_pct: float = 0.5,
    cache_file: Optional[str] = "cache/analysis_cache.json",
    force_new: bool = False
) -> AnalysisCache:
    """Get global analysis cache instance."""
    global _cache_instance
    
    if _cache_instance is None or force_new:
        # Ensure cache directory exists
        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        
        _cache_instance = AnalysisCache(
            max_age_minutes=max_age_minutes,
            price_threshold_pct=price_threshold_pct,
            cache_file=cache_file
        )
    
    return _cache_instance


if __name__ == "__main__":
    # Test the cache
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    cache = AnalysisCache(max_age_minutes=1, price_threshold_pct=0.5)
    
    # Simulate analysis
    analysis = {
        "ticker": "TSLA",
        "action": "BUY",
        "conviction": "HIGH",
        "plan": {"buy_zone": "$245-$250"}
    }
    
    # Store
    cache.set("TSLA", 245.50, analysis)
    
    # Hit (same price)
    result = cache.get("TSLA", 245.60)
    print(f"Hit 1: {result.get('_cached') if result else 'MISS'}")
    
    # Hit (small price change)
    result = cache.get("TSLA", 246.00)
    print(f"Hit 2: {result.get('_cached') if result else 'MISS'}")
    
    # Miss (price moved too much)
    result = cache.get("TSLA", 250.00)
    print(f"Miss (price): {result.get('_cached') if result else 'MISS'}")
    
    # Stats
    print(f"Stats: {cache.get_stats()}")

