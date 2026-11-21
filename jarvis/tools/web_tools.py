"""Public-web search, weather lookup, and browser helpers."""

from __future__ import annotations

import logging
import webbrowser
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urljoin

import requests

from jarvis.config import get_settings
from jarvis.network import validate_public_url

logger = logging.getLogger(__name__)
_USER_AGENT = "JarvisLocalAssistant/0.1 (public web retrieval)"
_MAX_PAGE_BYTES = 1_000_000
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, _attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            cleaned = " ".join(data.split())
            if cleaned:
                self.parts.append(cleaned)


def _validate_public_url(url: str, *, require_https: bool = False) -> str:
    return validate_public_url(url, require_https=require_https)


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the public web using DuckDuckGo's search results."""
    term = query.strip()
    if not term:
        raise ValueError("Search query cannot be empty.")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")
    from ddgs import DDGS

    results = list(DDGS().text(term, max_results=max_results))
    return {"ok": True, "data": {"query": term, "results": results}}


def get_weather(location: str | None = None) -> dict[str, Any]:
    """Look up current conditions using the free Open-Meteo geocoding and forecast APIs."""
    place = (location or get_settings().default_city or "").strip()
    if not place:
        raise ValueError(
            "Provide a city or set JARVIS_DEFAULT_CITY in the local .env file."
        )
    timeout = get_settings().request_timeout_seconds
    geocoding = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    geocoding.raise_for_status()
    locations = geocoding.json().get("results") or []
    if not locations:
        return {
            "ok": False,
            "error": {
                "code": "location_not_found",
                "message": f"No matching location was found for {place!r}.",
            },
        }

    match = locations[0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": match["latitude"],
            "longitude": match["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    forecast.raise_for_status()
    forecast_data = forecast.json()
    current = forecast_data.get("current")
    if not current:
        return {
            "ok": False,
            "error": {
                "code": "weather_unavailable",
                "message": "The weather service did not return current conditions.",
            },
        }
    return {
        "ok": True,
        "data": {
            "location": {
                "name": match.get("name"),
                "admin1": match.get("admin1"),
                "country": match.get("country"),
                "timezone": forecast_data.get("timezone"),
            },
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "conditions": _WEATHER_CODES.get(
                current.get("weather_code"), "unknown conditions"
            ),
            "observed_at": current.get("time"),
        },
    }


def open_url(url: str) -> dict[str, Any]:
    """Open a validated public HTTP(S) URL in the default browser."""
    safe_url = _validate_public_url(url)
    if not webbrowser.open(safe_url, new=2):
        return {
            "ok": False,
            "error": {
                "code": "browser_unavailable",
                "message": "The default browser could not be opened.",
            },
        }
    return {"ok": True, "data": {"url": safe_url}}


def open_browser_search(query: str) -> dict[str, Any]:
    """Open a DuckDuckGo search page in the default browser."""
    term = query.strip()
    if not term:
        raise ValueError("Search query cannot be empty.")
    url = f"https://duckduckgo.com/?q={quote_plus(term)}"
    if not webbrowser.open(url, new=2):
        return {
            "ok": False,
            "error": {
                "code": "browser_unavailable",
                "message": "The default browser could not be opened.",
            },
        }
    return {"ok": True, "data": {"url": url}}


def open_youtube_search(query: str) -> dict[str, Any]:
    """Open a YouTube search page in the default browser."""
    term = query.strip()
    if not term:
        raise ValueError("Search query cannot be empty.")
    url = f"https://www.youtube.com/results?search_query={quote_plus(term)}"
    if not webbrowser.open(url, new=2):
        return {
            "ok": False,
            "error": {
                "code": "browser_unavailable",
                "message": "The default browser could not be opened.",
            },
        }
    return {"ok": True, "data": {"url": url}}


def fetch_webpage(url: str, max_chars: int = 12_000) -> dict[str, Any]:
    """Fetch a public web page and return extracted readable text."""
    if not 500 <= max_chars <= 30_000:
        raise ValueError("max_chars must be between 500 and 30000.")
    current_url = _validate_public_url(url, require_https=True)
    timeout = get_settings().request_timeout_seconds
    body = bytearray()
    content_type = ""
    for _ in range(5):
        with requests.get(
            current_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        ) as response:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The web page returned a redirect without a target.")
                current_url = _validate_public_url(
                    urljoin(current_url, location), require_https=True
                )
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                raise ValueError(
                    f"Unsupported page content type: {content_type or 'unknown'}."
                )
            for chunk in response.iter_content(chunk_size=16_384):
                if not chunk:
                    continue
                remaining = _MAX_PAGE_BYTES - len(body)
                body.extend(chunk[:remaining])
                if len(body) >= _MAX_PAGE_BYTES:
                    break
            encoding = response.encoding or "utf-8"
            break
    else:
        raise ValueError("The web page redirected too many times.")

    decoded = bytes(body).decode(encoding, errors="replace")
    if "text/html" in content_type:
        extractor = _TextExtractor()
        extractor.feed(decoded)
        text = " ".join(extractor.parts)
    else:
        text = " ".join(decoded.split())
    return {
        "ok": True,
        "data": {
            "url": current_url,
            "text": text[:max_chars],
            "truncated": len(text) > max_chars or len(body) >= _MAX_PAGE_BYTES,
        },
    }


def summarize_webpage(url: str, max_chars: int = 1800) -> dict[str, Any]:
    """Return a short extractive summary of a public page's leading sentences."""
    if not 200 <= max_chars <= 5000:
        raise ValueError("max_chars must be between 200 and 5000.")
    page_result = fetch_webpage(url, max_chars=30_000)
    if not page_result.get("ok"):
        return page_result
    page_data = page_result["data"]
    text = str(page_data["text"])
    sentences = [sentence.strip() for sentence in text.replace("!", ".").split(".")]
    summary = ". ".join(sentence for sentence in sentences if sentence)[:max_chars]
    if summary and not summary.endswith((".", "!", "?")):
        summary += "…"
    return {
        "ok": True,
        "data": {
            "url": page_data["url"],
            "summary": summary or "The page did not contain readable text.",
            "method": "extractive",
        },
    }
