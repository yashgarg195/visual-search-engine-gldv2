"""Safe On-Demand Thumbnail Fetcher for Geometric Verification.

Fetches candidate images on-the-fly for Stage-2 verification with:
- Strict timeouts (1.5s) to prevent query stalls.
- Custom User-Agent conforming to Wikimedia Commons API policy.
- Local in-memory LRU cache to prevent redundant HTTP requests.
- Standard library urllib.request (zero extra dependencies).
- Fail-safe fallback: returns None on any network error or rate limit (HTTP 429/403).
"""

from collections import OrderedDict
import io
import threading
import urllib.error
import urllib.request
from typing import Optional, Union
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger("thumbnail_fetcher")

# Wikimedia requires a distinct user-agent with contact info
DEFAULT_USER_AGENT = "GLDv2-VisualSearchEngine/1.0 (https://github.com/visual-search; research@example.com)"


class ThumbnailFetcher:
    """Thread-safe LRU-cached thumbnail fetcher with strict rate and timeout guards."""

    def __init__(self, max_cache_size: int = 128, timeout: float = 1.5):
        self.max_cache_size = max_cache_size
        self.timeout = timeout
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()
        # The class is documented as thread-safe and is reached from FastAPI's
        # request threadpool, so every cache mutation must be serialized.
        self._lock = threading.Lock()

    def fetch(self, url: str) -> Optional[Image.Image]:
        """Fetch image bytes from URL and return as PIL RGB Image, or None on failure."""
        if not url or not url.startswith("http"):
            return None

        # Check cache
        with self._lock:
            if url in self._cache:
                self._cache.move_to_end(url)
                return self._cache[url]

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "image/webp,image/jpeg,image/png,*/*;q=0.8",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.debug("Thumbnail fetch failed for %s: HTTP %d", url, resp.status)
                    return None
                data = resp.read()

            img = Image.open(io.BytesIO(data)).convert("RGB")

            # Store in LRU cache
            with self._lock:
                if url not in self._cache and len(self._cache) >= self.max_cache_size:
                    self._cache.popitem(last=False)
                self._cache[url] = img
                self._cache.move_to_end(url)
            return img
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            logger.debug("Network error fetching thumbnail %s: %s", url, e)
            return None
        except Exception as e:
            logger.debug("Failed decoding image from %s: %s", url, e)
            return None


# Module-level singleton fetcher
_global_fetcher: Optional[ThumbnailFetcher] = None


def get_thumbnail_fetcher() -> ThumbnailFetcher:
    global _global_fetcher
    if _global_fetcher is None:
        _global_fetcher = ThumbnailFetcher()
    return _global_fetcher
