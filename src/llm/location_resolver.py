"""LLM Location & Disambiguation Resolver using Gemini 3.8 Flash.

Resolves ambiguous sub-landmark names (e.g. 'Monument to Minin and Pozharsky',
'Mehtab Bagh') to the primary iconic anchor attraction ('St. Basil's Cathedral',
'Taj Mahal'), providing geolocation coordinates, visitor context, and canonical search queries.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from src.config import load_env_file
load_env_file()

from src.utils.logging import get_logger

logger = get_logger("location_resolver")


from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_RESOLVER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini_resolver")

# Curated anchor knowledge for instant disambiguation of ambiguous sub-landmarks
PROMINENT_ANCHOR_MAP: Dict[str, Dict[str, Any]] = {
    "minin and pozharsky": {
        "prominent_attraction": "Saint Basil's Cathedral",
        "specific_subject": "Monument to Minin and Pozharsky",
        "city": "Moscow",
        "country": "Russia",
        "latitude": 55.7525,
        "longitude": 37.6231,
        "summary": "Located on Moscow's Red Square directly in front of the world-famous Saint Basil's Cathedral, this bronze statue commemorates Dmitry Pozharsky and Kuzma Minin, leaders of the 1612 resistance.",
        "search_query": "Saint Basil Cathedral Moscow",
    },
    "mehtab bagh": {
        "prominent_attraction": "Taj Mahal",
        "specific_subject": "Mehtab Bagh (Moonlight Garden)",
        "city": "Agra",
        "country": "India",
        "latitude": 27.1751,
        "longitude": 78.0421,
        "summary": "Positioned directly north across the Yamuna River from the Taj Mahal, Mehtab Bagh provides the quintessential panoramic vantage point for the iconic white marble wonder.",
        "search_query": "Taj Mahal Agra",
    },
    "bir-hakeim": {
        "prominent_attraction": "Eiffel Tower",
        "specific_subject": "Pont de Bir-Hakeim",
        "city": "Paris",
        "country": "France",
        "latitude": 48.8584,
        "longitude": 2.2945,
        "summary": "A celebrated steel bridge spanning the River Seine in Paris, globally renowned for framing cinematic vistas of the nearby Eiffel Tower.",
        "search_query": "Eiffel Tower Paris",
    },
    "arch of constantine": {
        "prominent_attraction": "Colosseum",
        "specific_subject": "Arch of Constantine",
        "city": "Rome",
        "country": "Italy",
        "latitude": 41.8902,
        "longitude": 12.4922,
        "summary": "Situated between the Roman Colosseum and Palatine Hill, this triumphal arch was dedicated in 315 AD to celebrate Emperor Constantine I's victory.",
        "search_query": "Colosseum Rome",
    },
    "pantheon": {
        "prominent_attraction": "Pantheon",
        "specific_subject": "Pantheon",
        "city": "Rome",
        "country": "Italy",
        "latitude": 41.8986,
        "longitude": 12.4769,
        "summary": "A former Roman temple and church famous for having the world's largest unreinforced concrete dome and oculus.",
        "search_query": "Pantheon Rome Italy",
    },
}


@dataclass
class LocationResolution:
    prominent_attraction: str
    specific_subject: str
    city: str
    country: str
    latitude: float
    longitude: float
    summary: str
    search_query: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocationResolver:
    """Disambiguates visual landmark search results using Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.8-flash", cache_path: str = "outputs/cache/landmark_intel_cache.json"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.client = None
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

        if self.api_key:
            try:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=10000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
                logger.info("Initialized Gemini client with model %s (disk cache loaded: %d entries)", self.model, len(self.cache))
            except Exception as e:
                logger.warning("Failed to initialize Gemini client: %s", e)

    def _load_cache(self) -> None:
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
        except Exception as e:
            logger.warning("Failed to load landmark cache from %s: %s", self.cache_path, e)
            self.cache = {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save landmark cache to %s: %s", self.cache_path, e)

    def is_available(self) -> bool:
        return self.client is not None

    def resolve(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        fallback_coords: Optional[Dict[str, float]] = None,
    ) -> Optional[LocationResolution]:
        """Resolve ambiguous or specific candidates to the prominent landmark anchor."""
        clean_q = re.sub(r"^(Text:\s*['\"]?|Image:\s*)", "", query_text or "").strip().rstrip("'\"").lower()
        top_name = (candidates[0].get("landmark_name") or candidates[0].get("title") or "") if candidates else ""
        cache_key = clean_q or top_name.lower().strip()

        # Check in-memory / persistent disk cache first
        if cache_key and cache_key in self.cache:
            logger.info("Resolved landmark from persistent disk cache for key: '%s' (0 API calls)", cache_key)
            return LocationResolution(**self.cache[cache_key])

        if not self.is_available():
            logger.info("Gemini API client not configured; using heuristic fallback.")
            res = self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)
            if res and cache_key:
                self.cache[cache_key] = res.to_dict()
                self._save_cache()
            return res

        # Extract context from top candidates
        cand_summaries = []
        valid_lat = 0.0
        valid_lon = 0.0

        for i, c in enumerate(candidates[:8]):
            title = c.get("landmark_name") or c.get("title") or c.get("name") or "Unknown"
            hier = c.get("hierarchical_label") or c.get("hierarchy") or ""
            lat = c.get("latitude")
            lon = c.get("longitude")
            if (not valid_lat or not valid_lon) and lat and lon:
                try:
                    valid_lat = float(lat)
                    valid_lon = float(lon)
                except (ValueError, TypeError):
                    pass
            score = c.get("similarity") or c.get("score") or 0.0
            cand_summaries.append(
                f"- Candidate {i+1}: '{title}' | Region: '{hier}' | Coords: ({lat}, {lon}) | Sim: {score:.3f}"
            )

        context_str = "\n".join(cand_summaries)

        prompt = f"""You are a world-class geographic and architectural intelligence system.
A user performed a visual landmark search. The visual retrieval engine found these top candidates:
{context_str}

User query context: '{query_text}'

YOUR TASK:
1. Identify the exact geographic location and where the user is (City, Country).
2. Identify the PRIMARY PROMINENT ATTRACTION / LANDMARK ANCHOR in that immediate area.
   CRITICAL DISAMBIGUATION RULE: If the visual candidates point to a specific statue, garden, gate, or plaza (e.g. 'Monument to Minin and Pozharsky' in Moscow, or 'Mehtab Bagh' across from Taj Mahal), do NOT simply repeat the obscure sub-monument name as the main attraction. The prominent attraction must be the world-famous iconic anchor that visitors and travelers recognize (e.g., 'St. Basil's Cathedral' or 'Taj Mahal').
3. Identify the 'specific_subject' if the photo is specifically focused on a sub-element (e.g. 'Monument to Minin and Pozharsky').
4. Provide the exact GPS coordinates (latitude, longitude) of this prominent attraction.
5. Provide an engaging, informative 2-sentence visitor summary explaining what the location is and the relationship between the specific subject and the main attraction.
6. Provide a concise 'search_query' (3-5 words) suitable for finding 3-5 high-resolution canonical reference photographs on Wikimedia / Google.

OUTPUT STRICT JSON ONLY MATCHING THIS SCHEMA:
{{
  "prominent_attraction": "string",
  "specific_subject": "string",
  "city": "string",
  "country": "string",
  "latitude": float,
  "longitude": float,
  "summary": "string",
  "search_query": "string"
}}
"""

        def _call_gemini():
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

        try:
            future = _RESOLVER_EXECUTOR.submit(_call_gemini)
            response = future.result(timeout=2.0)

            raw_text = response.text.strip()
            data = json.loads(raw_text)

            lat_val = float(data.get("latitude", 0.0)) or valid_lat
            lon_val = float(data.get("longitude", 0.0)) or valid_lon

            resolution = LocationResolution(
                prominent_attraction=data.get("prominent_attraction", "Unknown Landmark"),
                specific_subject=data.get("specific_subject", data.get("prominent_attraction", "")),
                city=data.get("city", ""),
                country=data.get("country", ""),
                latitude=lat_val,
                longitude=lon_val,
                summary=data.get("summary", ""),
                search_query=data.get("search_query", data.get("prominent_attraction", "")),
            )
            if cache_key:
                self.cache[cache_key] = resolution.to_dict()
                self._save_cache()
            return resolution
        except FutureTimeoutError:
            logger.warning("Gemini location resolver exceeded 2.0s; using instant local fallback.")
            res = self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)
            if res and cache_key:
                self.cache[cache_key] = res.to_dict()
                self._save_cache()
            return res
        except Exception as e:
            logger.warning("Error resolving location with Gemini: %s. Falling back to heuristic.", e)
            res = self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)
            if res and cache_key:
                self.cache[cache_key] = res.to_dict()
                self._save_cache()
            return res

    def _heuristic_fallback(
        self,
        candidates: List[Dict[str, Any]],
        fallback_coords: Optional[Dict[str, float]] = None,
        query_text: str = "",
    ) -> Optional[LocationResolution]:
        """Rule-based fallback when LLM is unavailable or times out."""
        if not candidates and not query_text:
            return None

        top = candidates[0] if candidates else {}
        raw_title = top.get("landmark_name") or top.get("title") or top.get("name") or query_text or "Unknown Landmark"
        title = raw_title.replace("_", " ").strip()
        hier = top.get("hierarchical_label") or top.get("hierarchy") or ""
        parts = [p.strip().replace("_", " ") for p in hier.split(">") if p.strip()]

        country = parts[0] if len(parts) > 0 else ""
        city = parts[-1] if len(parts) > 1 else ""

        # Check curated prominent anchor knowledge base against query_text and candidate titles
        combined_text = f"{query_text} {title}".lower()
        for anchor_key, anchor_data in PROMINENT_ANCHOR_MAP.items():
            if anchor_key in combined_text or any(anchor_key in (c.get("landmark_name", "") or "").lower() for c in candidates[:4]):
                return LocationResolution(
                    prominent_attraction=anchor_data["prominent_attraction"],
                    specific_subject=anchor_data["specific_subject"],
                    city=anchor_data["city"],
                    country=anchor_data["country"],
                    latitude=float(anchor_data["latitude"]),
                    longitude=float(anchor_data["longitude"]),
                    summary=anchor_data["summary"],
                    search_query=anchor_data["search_query"],
                )

        lat = 0.0
        lon = 0.0
        for c in candidates:
            c_lat = c.get("latitude")
            c_lon = c.get("longitude")
            if c_lat and c_lon:
                try:
                    lat = float(c_lat)
                    lon = float(c_lon)
                    break
                except (ValueError, TypeError):
                    pass

        if not lat and fallback_coords:
            lat = fallback_coords.get("latitude", 0.0)
            lon = fallback_coords.get("longitude", 0.0)

        # If coordinates or country are still missing or invalid, resolve via Nominatim
        if (lat == 0.0 or lon == 0.0 or country in ("", "nan")):
            try:
                import urllib.parse, urllib.request
                clean_q = re.sub(r"^(Text:\s*['\"]?|Image:\s*)", "", query_text or "").strip().rstrip("'\"")
                geo_query = clean_q if clean_q else title
                geo_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(geo_query)}&format=json&limit=1"
                geo_req = urllib.request.Request(geo_url, headers={"User-Agent": "VisualLandmarkSearch/1.0"})
                with urllib.request.urlopen(geo_req, timeout=1.5) as geo_resp:
                    geo_json = json.loads(geo_resp.read().decode())
                    if geo_json:
                        first_geo = geo_json[0]
                        lat = float(first_geo.get("lat", lat))
                        lon = float(first_geo.get("lon", lon))
                        disp = first_geo.get("display_name", "")
                        disp_parts = [p.strip() for p in disp.split(",") if p.strip()]
                        if disp_parts:
                            country = disp_parts[-1]
                            if len(disp_parts) >= 3 and not city:
                                city = disp_parts[-3]
            except Exception as geo_err:
                logger.debug("Geocoding lookup failed: %s", geo_err)

        if country == "nan":
            country = ""
        loc_desc = f"{city}, {country}" if city and country else (city or country or "")
        return LocationResolution(
            prominent_attraction=title,
            specific_subject=title,
            city=city,
            country=country,
            latitude=float(lat),
            longitude=float(lon),
            summary=f"{title} is a world-renowned landmark located in {loc_desc}." if loc_desc else f"{title} is a notable landmark.",
            search_query=f"{title} {city}".strip(),
        )
