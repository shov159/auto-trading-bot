"""
Rate Limiter - Global LLM Request Rate Control
===============================================
Ensures stable API usage with:
- Global rate limiting (min interval between calls)
- Single-flight locking (only 1 concurrent LLM request)
- Jitter to prevent thundering herd
- Request deduplication
"""
import threading
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Set
from dataclasses import dataclass

from src.logger import log_debug, log_warn, log_info


@dataclass
class RequestInfo:
    """Track info about an in-flight or recent request."""
    ticker: str
    started_at: datetime
    request_id: str


class RateLimiter:
    """
    Thread-safe rate limiter with jitter.
    
    Ensures minimum interval between LLM API calls with random jitter
    to prevent synchronized retry storms.
    """
    
    def __init__(self, min_interval_seconds: float = 5.0, jitter_pct: float = 0.15):
        """
        Args:
            min_interval_seconds: Minimum seconds between API calls
            jitter_pct: Random jitter as percentage of interval (+/- this %)
        """
        self.min_interval = min_interval_seconds
        self.jitter_pct = jitter_pct
        self._lock = threading.Lock()
        self._last_call_time: Optional[float] = None
    
    def _calculate_wait_with_jitter(self) -> float:
        """Calculate wait time with random jitter."""
        if self._last_call_time is None:
            return 0.0
        
        elapsed = time.time() - self._last_call_time
        base_wait = max(0, self.min_interval - elapsed)
        
        if base_wait > 0:
            # Add jitter: +/- jitter_pct of the interval
            jitter_range = self.min_interval * self.jitter_pct
            jitter = random.uniform(-jitter_range, jitter_range)
            wait_time = max(0, base_wait + jitter)
            return wait_time
        
        return 0.0
    
    def acquire(self) -> float:
        """
        Acquire rate limit slot. Blocks until rate limit allows.
        
        Returns:
            The actual wait time in seconds (0 if no wait needed)
        """
        with self._lock:
            wait_time = self._calculate_wait_with_jitter()
            
            if wait_time > 0:
                log_debug(f"Rate limiter: waiting {wait_time:.2f}s (interval={self.min_interval}s)")
                time.sleep(wait_time)
            
            self._last_call_time = time.time()
            return wait_time
    
    def reset(self):
        """Reset the rate limiter (useful after long idle periods)."""
        with self._lock:
            self._last_call_time = None


class SingleFlightLock:
    """
    Ensures only ONE LLM request runs at any time globally.
    Uses a threading lock to serialize all API calls.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current_request: Optional[RequestInfo] = None
    
    def acquire(self, ticker: str) -> str:
        """
        Acquire the single-flight lock.
        
        Args:
            ticker: The ticker being analyzed (for logging)
            
        Returns:
            A unique request_id for this request
        """
        request_id = str(uuid.uuid4())[:8]
        
        log_debug(f"[{request_id}] Waiting for LLM lock (ticker: {ticker})...")
        self._lock.acquire()
        
        self._current_request = RequestInfo(
            ticker=ticker,
            started_at=datetime.now(),
            request_id=request_id
        )
        
        log_debug(f"[{request_id}] LLM lock acquired for {ticker}")
        return request_id
    
    def release(self, request_id: str):
        """Release the single-flight lock."""
        if self._current_request and self._current_request.request_id == request_id:
            duration = (datetime.now() - self._current_request.started_at).total_seconds()
            log_debug(f"[{request_id}] LLM lock released (duration: {duration:.2f}s)")
            self._current_request = None
        
        try:
            self._lock.release()
        except RuntimeError:
            pass  # Lock wasn't held
    
    def is_locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.locked()
    
    def get_current_request(self) -> Optional[RequestInfo]:
        """Get info about the currently running request."""
        return self._current_request


class TickerDeduplicator:
    """
    Prevents duplicate analysis of the same ticker within a time window.
    """
    
    def __init__(self, cooldown_seconds: float = 120.0):
        """
        Args:
            cooldown_seconds: Minimum seconds before same ticker can be analyzed again
        """
        self.cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._recent_requests: Dict[str, datetime] = {}
        self._inflight: Set[str] = set()
    
    def is_duplicate(self, ticker: str) -> bool:
        """
        Check if this ticker was recently analyzed or is currently in-flight.
        
        Returns:
            True if this is a duplicate request that should be skipped
        """
        with self._lock:
            now = datetime.now()
            
            # Check if in-flight
            if ticker in self._inflight:
                log_warn(f"Duplicate blocked: {ticker} is currently being analyzed")
                return True
            
            # Check cooldown
            if ticker in self._recent_requests:
                last_request = self._recent_requests[ticker]
                elapsed = (now - last_request).total_seconds()
                
                if elapsed < self.cooldown:
                    remaining = self.cooldown - elapsed
                    log_warn(f"Duplicate blocked: {ticker} analyzed {elapsed:.0f}s ago (cooldown: {remaining:.0f}s remaining)")
                    return True
            
            return False
    
    def mark_started(self, ticker: str):
        """Mark a ticker as currently being analyzed."""
        with self._lock:
            self._inflight.add(ticker)
            log_debug(f"Marked {ticker} as in-flight")
    
    def mark_completed(self, ticker: str):
        """Mark a ticker analysis as completed."""
        with self._lock:
            self._inflight.discard(ticker)
            self._recent_requests[ticker] = datetime.now()
            log_debug(f"Marked {ticker} as completed")
    
    def clear_inflight(self, ticker: str):
        """Remove a ticker from in-flight (on error)."""
        with self._lock:
            self._inflight.discard(ticker)
    
    def get_inflight(self) -> Set[str]:
        """Get set of currently in-flight tickers."""
        with self._lock:
            return self._inflight.copy()
    
    def cleanup_old_entries(self, max_age_seconds: float = 300.0):
        """Remove entries older than max_age from recent_requests."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=max_age_seconds)
            
            old_tickers = [
                ticker for ticker, ts in self._recent_requests.items()
                if ts < cutoff
            ]
            
            for ticker in old_tickers:
                del self._recent_requests[ticker]


class ScanLock:
    """
    Prevents overlapping watchlist scans.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._is_scanning = False
        self._scan_started: Optional[datetime] = None
        self._scan_type: Optional[str] = None
    
    def try_acquire(self, scan_type: str = "manual") -> bool:
        """
        Try to acquire the scan lock.
        
        Returns:
            True if lock acquired, False if another scan is running
        """
        with self._lock:
            if self._is_scanning:
                elapsed = (datetime.now() - self._scan_started).total_seconds() if self._scan_started else 0
                log_warn(f"Scan blocked: {self._scan_type} scan already running for {elapsed:.0f}s")
                return False
            
            self._is_scanning = True
            self._scan_started = datetime.now()
            self._scan_type = scan_type
            log_info(f"Scan lock acquired: {scan_type}")
            return True
    
    def release(self):
        """Release the scan lock."""
        with self._lock:
            if self._is_scanning and self._scan_started:
                duration = (datetime.now() - self._scan_started).total_seconds()
                log_info(f"Scan lock released: {self._scan_type} (duration: {duration:.1f}s)")
            
            self._is_scanning = False
            self._scan_started = None
            self._scan_type = None
    
    def is_scanning(self) -> bool:
        """Check if a scan is currently running."""
        with self._lock:
            return self._is_scanning
    
    def get_scan_info(self) -> Optional[Dict]:
        """Get info about the current scan."""
        with self._lock:
            if not self._is_scanning:
                return None
            
            return {
                "type": self._scan_type,
                "started": self._scan_started,
                "duration": (datetime.now() - self._scan_started).total_seconds() if self._scan_started else 0
            }


# =============================================================================
# GLOBAL INSTANCES (Singleton pattern)
# =============================================================================

_rate_limiter: Optional[RateLimiter] = None
_single_flight: Optional[SingleFlightLock] = None
_deduplicator: Optional[TickerDeduplicator] = None
_scan_lock: Optional[ScanLock] = None


def get_rate_limiter(min_interval: float = 5.0) -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(min_interval_seconds=min_interval)
    return _rate_limiter


def get_single_flight() -> SingleFlightLock:
    """Get global single-flight lock instance."""
    global _single_flight
    if _single_flight is None:
        _single_flight = SingleFlightLock()
    return _single_flight


def get_deduplicator(cooldown: float = 120.0) -> TickerDeduplicator:
    """Get global ticker deduplicator instance."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = TickerDeduplicator(cooldown_seconds=cooldown)
    return _deduplicator


def get_scan_lock() -> ScanLock:
    """Get global scan lock instance."""
    global _scan_lock
    if _scan_lock is None:
        _scan_lock = ScanLock()
    return _scan_lock


if __name__ == "__main__":
    # Test the rate limiter
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    limiter = RateLimiter(min_interval_seconds=2.0, jitter_pct=0.15)
    
    print("Testing rate limiter...")
    for i in range(5):
        wait = limiter.acquire()
        print(f"Request {i+1}: waited {wait:.3f}s")

