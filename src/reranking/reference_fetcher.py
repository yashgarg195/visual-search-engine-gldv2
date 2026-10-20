"""Canonical Reference Image Fetcher for Disambiguated Landmarks.

Fetches 3-5 pristine, high-resolution photographic reference images of global landmarks
using the Wikimedia Commons API and Wikipedia REST endpoints.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from src.utils.logging import get_logger

logger = get_logger("reference_fetcher")

HEADERS = {
    "User-Agent": "VisualSearchEngine/1.0 (https://github.com/visual-search; educational/research) python-requests"
}


class ReferenceImageFetcher:
    """Fetches canonical photographic reference images for landmarks."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 6.0):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/references")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._memory_cache: Dict[str, List[Dict[str, Any]]] = {}

    def fetch_reference_images(self, search_query: str, max_images: int = 4) -> List[Dict[str, Any]]:
        """Fetch 3-5 high-resolution canonical reference images for a landmark search query."""
        clean_query = search_query.strip()
        if not clean_query:
            return []

        # Check memory cache
        if clean_query.lower() in self._memory_cache:
            return self._memory_cache[clean_query.lower()]

        results: List[Dict[str, Any]] = []

        # 1. Search Wikimedia Commons (File namespace = 6)
        try:
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{clean_query} -map -plan -diagram -icon -svg",
                "gsrnamespace": 6,
                "gsrlimit": max(max_images * 2, 8),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": 640,
                "format": "json",
            }
            resp = requests.get(url, params=params, headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for pid, page in pages.items():
                    info_list = page.get("imageinfo", [])
                    if not info_list:
                        continue
                    info = info_list[0]
                    mime = info.get("mime", "").lower()
                    # Filter for photographic images only (JPEG / PNG), skip SVGs / PDFs / TIF
                    if "image/jpeg" not in mime and "image/png" not in mime and "image/webp" not in mime:
                        continue

                    title = page.get("title", "").replace("File:", "").replace("_", " ")
                    thumb_url = info.get("thumburl") or info.get("url")
                    orig_url = info.get("url")

                    if thumb_url:
                        results.append({
                            "title": title,
                            "thumb_url": thumb_url,
                            "original_url": orig_url,
                            "source": "Wikimedia Commons",
                        })
                        if len(results) >= max_images:
                            break
        except Exception as e:
            logger.warning("Wikimedia search failed for '%s': %s", clean_query, e)

        # 2. Fallback to Wikipedia Page Images if Commons returned fewer than 2 images
        if len(results) < 2:
            try:
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean_query)}"
                resp = requests.get(wiki_url, headers=HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    orig_img = data.get("originalimage", {})
                    thumb_img = data.get("thumbnail", {})
                    if thumb_img.get("source"):
                        results.insert(0, {
                            "title": data.get("title", clean_query),
                            "thumb_url": thumb_img.get("source"),
                            "original_url": orig_img.get("source", thumb_img.get("source")),
                            "source": "Wikipedia",
                        })
            except Exception as e:
                logger.warning("Wikipedia summary image fallback failed for '%s': %s", clean_query, e)

        # Cache results in memory
        self._memory_cache[clean_query.lower()] = results
        logger.info("Fetched %d canonical reference images for query '%s'", len(results), clean_query)
        return results

    def download_image_to_cache(self, image_url: str) -> Optional[Path]:
        """Download an image URL to local cache for feature extraction."""
        try:
            url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
            ext = ".jpg"
            if ".png" in image_url.lower():
                ext = ".png"
            elif ".webp" in image_url.lower():
                ext = ".webp"
            local_path = self.cache_dir / f"{url_hash}{ext}"

            if local_path.is_file() and local_path.stat().st_size > 1024:
                return local_path

            resp = requests.get(image_url, headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200 and len(resp.content) > 1024:
                local_path.write_bytes(resp.content)
                return local_path
        except Exception as e:
            logger.warning("Failed downloading reference image %s: %s", image_url, e)
        return None

    def download_images_parallel(self, urls: List[str]) -> Dict[str, Optional[Path]]:
        """Concurrently download reference images in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        res = {}
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as ex:
            fut_map = {ex.submit(self.download_image_to_cache, u): u for u in urls if u}
            for fut in fut_map:
                u = fut_map[fut]
                try:
                    res[u] = fut.result()
                except Exception:
                    res[u] = None
        return res
