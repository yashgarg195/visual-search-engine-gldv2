"""LLM Location & Disambiguation Resolver using Gemini 3.8 Flash.

Resolves ambiguous sub-landmark names (e.g. 'Monument to Minin and Pozharsky',
'Mehtab Bagh') to the primary iconic anchor attraction ('St. Basil's Cathedral',
'Taj Mahal'), providing geolocation coordinates, visitor context, and canonical search queries.
"""

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from src.config import load_env_file
load_env_file()

from src.utils.logging import get_logger

logger = get_logger("location_resolver")

# The Gemini SDK is an optional dependency: the search engine is fully
# functional without LLM enrichment, so a missing package must not take down
# the whole API at import time.
try:
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:  # pragma: no cover - depends on local install
    genai = None
    types = None
    HAS_GENAI = False
    logger.warning(
        "google-genai is not installed; landmark LLM enrichment will be disabled."
    )


from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

_RESOLVER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini_resolver")

# Bound the persistent resolution cache so a long-running server cannot grow it
# without limit (it is rewritten to disk in full on every miss).
MAX_CACHE_ENTRIES = 5000

# Curated anchor knowledge for instant disambiguation of ambiguous sub-landmarks
PROMINENT_ANCHOR_MAP: Dict[str, Dict[str, Any]] = {
    "gateway of india": {
        "prominent_attraction": "Gateway of India",
        "specific_subject": "Gateway of India",
        "city": "Mumbai",
        "country": "India",
        "latitude": 18.9220,
        "longitude": 72.8347,
        "summary": "Built in the early 20th century overlooking the Arabian Sea, the Gateway of India is an iconic Indo-Saracenic arch-monument and the premier historic symbol of Mumbai.",
        "search_query": "Gateway of India Mumbai",
    },
    "eiffel tower": {
        "prominent_attraction": "Eiffel Tower",
        "specific_subject": "Eiffel Tower",
        "city": "Paris",
        "country": "France",
        "latitude": 48.8584,
        "longitude": 2.2945,
        "summary": "Constructed in 1889 for the Exposition Universelle, the Eiffel Tower is the quintessential global icon of Paris and France, standing 330 meters tall on the Champ de Mars.",
        "search_query": "Eiffel Tower Paris France",
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
    "taj mahal": {
        "prominent_attraction": "Taj Mahal",
        "specific_subject": "Taj Mahal",
        "city": "Agra",
        "country": "India",
        "latitude": 27.1751,
        "longitude": 78.0421,
        "summary": "An ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, commissioned in 1631 by Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.",
        "search_query": "Taj Mahal Agra India",
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
    "saint basil": {
        "prominent_attraction": "Saint Basil's Cathedral",
        "specific_subject": "Saint Basil's Cathedral",
        "city": "Moscow",
        "country": "Russia",
        "latitude": 55.7525,
        "longitude": 37.6231,
        "summary": "Commissioned by Ivan the Terrible in the 16th century on Moscow's Red Square, Saint Basil's Cathedral is globally famous for its fairy-tale cluster of multi-colored onion domes.",
        "search_query": "Saint Basil Cathedral Moscow",
    },
    "colosseum": {
        "prominent_attraction": "Colosseum",
        "specific_subject": "Colosseum",
        "city": "Rome",
        "country": "Italy",
        "latitude": 41.8902,
        "longitude": 12.4922,
        "summary": "The largest ancient amphitheatre ever built, the Colosseum stands in the center of Rome as the ultimate monument of Imperial Roman engineering and gladiatorial spectacles.",
        "search_query": "Colosseum Rome Italy",
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
    "india gate": {
        "prominent_attraction": "India Gate",
        "specific_subject": "India Gate",
        "city": "New Delhi",
        "country": "India",
        "latitude": 28.6129,
        "longitude": 77.2295,
        "summary": "Located on the Rajpath in New Delhi, India Gate is a majestic 42-meter triumphal arch war memorial dedicated to Indian soldiers of the First World War.",
        "search_query": "India Gate New Delhi",
    },
    "space needle": {
        "prominent_attraction": "Space Needle",
        "specific_subject": "Space Needle",
        "city": "Seattle",
        "country": "United States",
        "latitude": 47.6205,
        "longitude": -122.3493,
        "summary": "Standing 184 meters tall at the Seattle Center, the Space Needle is a futuristic observation tower built for the 1962 World's Fair and an icon of the Pacific Northwest.",
        "search_query": "Space Needle Seattle",
    },
    "spilled blood": {
        "prominent_attraction": "Church of the Savior on Spilled Blood",
        "specific_subject": "Church of the Savior on Spilled Blood",
        "city": "Saint Petersburg",
        "country": "Russia",
        "latitude": 59.9401,
        "longitude": 30.3289,
        "summary": "An architectural masterpiece along the Griboyedov Canal in Saint Petersburg, renowned for its elaborate Russian Revival onion domes and over 7,500 square meters of mosaics.",
        "search_query": "Church Savior on Spilled Blood Saint Petersburg",
    },
    "saviour on the blood": {
        "prominent_attraction": "Church of the Savior on Spilled Blood",
        "specific_subject": "Church of the Savior on Spilled Blood",
        "city": "Saint Petersburg",
        "country": "Russia",
        "latitude": 59.9401,
        "longitude": 30.3289,
        "summary": "An architectural masterpiece along the Griboyedov Canal in Saint Petersburg, renowned for its elaborate Russian Revival onion domes and over 7,500 square meters of mosaics.",
        "search_query": "Church Savior on Spilled Blood Saint Petersburg",
    },
    "grand canyon": {
        "prominent_attraction": "Grand Canyon",
        "specific_subject": "Grand Canyon National Park",
        "city": "Arizona",
        "country": "United States",
        "latitude": 36.0544,
        "longitude": -112.1401,
        "summary": "Carved over millions of years by the Colorado River in northern Arizona, the Grand Canyon is one of the world's most spectacular geological natural wonders.",
        "search_query": "Grand Canyon Arizona USA",
    },
    "christ the redeemer": {
        "prominent_attraction": "Christ the Redeemer",
        "specific_subject": "Christ the Redeemer (Cristo Redentor)",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "latitude": -22.9519,
        "longitude": -43.2105,
        "summary": "Standing 30 meters tall atop the 700-meter Corcovado mountain overlooking Rio de Janeiro, Christ the Redeemer is a colossal Art Deco statue of Jesus Christ and one of the New Seven Wonders of the World.",
        "search_query": "Christ the Redeemer Rio de Janeiro",
    },
    "cristo redentor": {
        "prominent_attraction": "Christ the Redeemer",
        "specific_subject": "Christ the Redeemer (Cristo Redentor)",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "latitude": -22.9519,
        "longitude": -43.2105,
        "summary": "Standing 30 meters tall atop the 700-meter Corcovado mountain overlooking Rio de Janeiro, Christ the Redeemer is a colossal Art Deco statue of Jesus Christ and one of the New Seven Wonders of the World.",
        "search_query": "Christ the Redeemer Rio de Janeiro",
    },
    "corcovado": {
        "prominent_attraction": "Christ the Redeemer",
        "specific_subject": "Corcovado Mountain & Christ the Redeemer",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "latitude": -22.9519,
        "longitude": -43.2105,
        "summary": "Rising dramatically above the Tijuca Forest in Rio de Janeiro, Mount Corcovado is world-renowned for crowning the monumental Art Deco statue of Christ the Redeemer.",
        "search_query": "Christ the Redeemer Corcovado Rio",
    },
    "cristo rei": {
        "prominent_attraction": "Christ the Redeemer",
        "specific_subject": "Christ the Redeemer / Sanctuary of Christ the King",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "latitude": -22.9519,
        "longitude": -43.2105,
        "summary": "Inspired by the world-famous Christ the Redeemer monument in Rio de Janeiro, this towering statue represents one of the iconic devotional representations of Christ overlooking the city.",
        "search_query": "Christ the Redeemer Rio de Janeiro",
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

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.1-flash-lite", cache_path: str = "outputs/cache/landmark_intel_cache.json"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = os.environ.get("GEMINI_MODEL", model)
        self.client = None
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._load_cache()

        if self.api_key and HAS_GENAI:
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

    @staticmethod
    def _is_degraded_cache_entry(entry: Dict[str, Any]) -> bool:
        """Check if an entry is a low-quality legacy fallback placeholder that should not be served."""
        summary = str(entry.get("summary", "")).lower()
        degraded_markers = [
            "notable landmark.",
            "located in tower.",
            "located in church.",
            "located in bridge.",
            "located in city.",
            "located in nan.",
            "is a notable landmark",
        ]
        return any(m in summary for m in degraded_markers)

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
            # Snapshot under the lock so we never serialize a dict that another
            # request thread is mutating, then write via a temp file + rename so
            # a crash mid-write cannot leave truncated JSON behind.
            with self._cache_lock:
                snapshot = dict(self.cache)
            tmp_path = f"{self.cache_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.cache_path)
        except Exception as e:
            logger.warning("Failed to save landmark cache to %s: %s", self.cache_path, e)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        """Store a resolution and bound the cache so it cannot grow forever."""
        with self._cache_lock:
            self.cache[key] = value
            if len(self.cache) > MAX_CACHE_ENTRIES:
                for stale in list(self.cache)[: len(self.cache) - MAX_CACHE_ENTRIES]:
                    self.cache.pop(stale, None)
        self._save_cache()

    def is_available(self) -> bool:
        return self.client is not None

    def resolve(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        fallback_coords: Optional[Dict[str, float]] = None,
    ) -> Optional[LocationResolution]:
        """Resolve ambiguous or specific candidates to the prominent landmark anchor."""
        raw_q = query_text or ""
        # Image queries arrive as "Image: <filename>". The filename is
        # attacker/user-controlled and carries no landmark meaning, so keying
        # the cache on it makes two unrelated photos named "photo.jpg" collide
        # and return each other's resolved landmark. Key image queries on the
        # retrieved landmark instead.
        is_image_query = bool(re.match(r"^\s*Image:\s*", raw_q))
        clean_q = re.sub(r"^(Text:\s*['\"]?|Image:\s*)", "", raw_q).strip().rstrip("'\"").lower()
        top_name = (candidates[0].get("landmark_name") or candidates[0].get("title") or "") if candidates else ""
        top_id = str(candidates[0].get("landmark_id") or "") if candidates else ""
        if is_image_query:
            cache_key = f"lm:{top_id}" if top_id else top_name.lower().strip()
        else:
            cache_key = clean_q or top_name.lower().strip()

        # Check in-memory / persistent disk cache first (purging legacy degraded entries)
        if cache_key and cache_key in self.cache:
            entry = self.cache[cache_key]
            if not self._is_degraded_cache_entry(entry):
                logger.info("Resolved landmark from persistent disk cache for key: '%s' (0 API calls)", cache_key)
                return LocationResolution(**entry)
            else:
                logger.info("Ignoring and evicting degraded legacy cache entry for key '%s'", cache_key)
                with self._cache_lock:
                    self.cache.pop(cache_key, None)

        # Check curated prominent anchor knowledge base first
        # Match strictly against the query text (for text queries) or the lead candidate (candidates[0]).
        # Never match secondary candidates (candidates[1..4]), which causes identity leakage between distinct landmarks.
        target_texts = [top_name.lower()]
        if not is_image_query and clean_q:
            target_texts.append(clean_q)

        for anchor_key, anchor_data in PROMINENT_ANCHOR_MAP.items():
            if any(anchor_key in t for t in target_texts if t):
                res = LocationResolution(
                    prominent_attraction=anchor_data["prominent_attraction"],
                    specific_subject=anchor_data["specific_subject"],
                    city=anchor_data["city"],
                    country=anchor_data["country"],
                    latitude=float(anchor_data["latitude"]),
                    longitude=float(anchor_data["longitude"]),
                    summary=anchor_data["summary"],
                    search_query=anchor_data["search_query"],
                )
                if cache_key:
                    self._cache_put(cache_key, res.to_dict())
                return res

        if not self.is_available():
            logger.info("Gemini API client not configured; using heuristic fallback.")
            return self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)

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
            # Try primary model, fallback to 3.5-flash or 3.8-flash if primary is unavailable
            models_to_try = [self.model, "gemini-3.5-flash", "gemini-3.8-flash"]
            seen = set()
            models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

            last_err = None
            for m in models_to_try:
                try:
                    return self.client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                except Exception as err:
                    last_err = err
                    logger.debug("Gemini model %s call failed: %s; trying next model", m, err)
            raise last_err or RuntimeError("No Gemini models succeeded")

        try:
            future = _RESOLVER_EXECUTOR.submit(_call_gemini)
            response = future.result(timeout=5.5)

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
                self._cache_put(cache_key, resolution.to_dict())
            return resolution
        except FutureTimeoutError:
            logger.warning("Gemini location resolver exceeded timeout; using instant local fallback.")
            return self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)
        except Exception as e:
            logger.warning("Error resolving location with Gemini: %s. Falling back to heuristic.", e)
            return self._heuristic_fallback(candidates, fallback_coords, query_text=query_text)

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

        # In GLDv2, hierarchical_label is an architectural/structure category (e.g. "tower", "church", "bridge")
        raw_cat = str(top.get("hierarchical_label") or top.get("hierarchy") or "").strip()
        category = "" if raw_cat.lower() in ("nan", "none", "unknown", "") else raw_cat.replace("_", " ")

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

        city = ""
        country = ""

        # Determine clean query for Nominatim geocoding
        # If image query or filename, NEVER use the uploaded filename as geocode search query!
        clean_q = re.sub(r"^(Text:\s*['\"]?|Image:\s*)", "", query_text or "").strip().rstrip("'\"")
        is_filename = bool(re.search(r"\.(jpe?g|png|webp|gif|bmp)$", clean_q, re.IGNORECASE))
        geo_query = title if (is_filename or not clean_q) else clean_q

        if (lat == 0.0 or lon == 0.0 or not country):
            try:
                import urllib.parse, urllib.request
                geo_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(geo_query)}&format=json&limit=1"
                geo_req = urllib.request.Request(geo_url, headers={"User-Agent": "VisualLandmarkSearch/2.0"})
                with urllib.request.urlopen(geo_req, timeout=2.0) as geo_resp:
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

        loc_desc = f"{city}, {country}" if city and country else (city or country or "")
        cat_desc = f" {category}" if category else " landmark"

        if loc_desc:
            summary = f"{title} is a renowned{cat_desc} located in {loc_desc}."
        else:
            summary = f"{title} is an iconic historic{cat_desc}."

        return LocationResolution(
            prominent_attraction=title,
            specific_subject=title,
            city=city,
            country=country,
            latitude=float(lat),
            longitude=float(lon),
            summary=summary,
            search_query=f"{title} {city}".strip(),
        )
