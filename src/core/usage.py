from __future__ import annotations

import time
import logging
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)

class UsageTracker:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.history: deque[tuple[float, int]] = deque()
        self.current_total = 0
        self.lock = Lock()

    def configure(self, limit: int, window_seconds: int) -> None:
        """Reconfigure tracker limits at runtime."""
        with self.lock:
            self.limit = int(limit)
            self.window_seconds = int(window_seconds)
            now = time.time()
            self._cleanup(now)

    def add_usage(self, count: int) -> bool:
        """Adds usage and returns True if within limit, False otherwise."""
        now = time.time()
        with self.lock:
            self._cleanup(now)
            
            if self.current_total + count > self.limit:
                logger.warning(
                    "Rate limit exceeded: current %d, requested %d, limit %d",
                    self.current_total,
                    count,
                    self.limit,
                )
                return False
            
            self.history.append((now, count))
            self.current_total += count
            return True

    def get_current_usage(self) -> int:
        now = time.time()
        with self.lock:
            self._cleanup(now)
            return self.current_total

    def _cleanup(self, now: float):
        while self.history and self.history[0][0] < now - self.window_seconds:
            _, amount = self.history.popleft()
            self.current_total -= amount
        if self.current_total < 0:
            self.current_total = 0

# Global usage tracker instance (can be configured via config later)
# Default: 1,000,000 characters per minute
global_usage_tracker = UsageTracker(limit=1_000_000, window_seconds=60)
