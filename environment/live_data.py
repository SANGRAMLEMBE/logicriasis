"""
Live data connectors for LogiCrisis.
Sources:
  1. OpenWeatherMap  — weather disruptions      (1000 free calls/day; set OPENWEATHERMAP_API_KEY)
  2. ExchangeRate-API — tariff/currency shocks  (free, no key needed)
  3. GDELT 2.0 Doc    — geopolitical conflicts  (free, no key needed)

Every connector returns the same disruption dict format and falls back to
synthetic data on any error so the simulation never crashes without live keys.
"""
from __future__ import annotations
import os, random, datetime
from typing import Any

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── City catalogue (matches world.py topology) ────────────────────────────────

_CITIES: dict[str, dict[str, float]] = {
    "Mumbai":    {"lat": 19.08, "lon": 72.88},
    "Delhi":     {"lat": 28.70, "lon": 77.10},
    "Kolkata":   {"lat": 22.57, "lon": 88.36},
    "Chennai":   {"lat": 13.08, "lon": 80.27},
    "Bangalore": {"lat": 12.97, "lon": 77.59},
    "Hyderabad": {"lat": 17.38, "lon": 78.49},
    "Pune":      {"lat": 18.52, "lon": 73.86},
    "Ahmedabad": {"lat": 23.03, "lon": 72.59},
    "Jaipur":    {"lat": 26.91, "lon": 75.79},
    "Surat":     {"lat": 21.17, "lon": 72.83},
}

# City → route IDs that pass through it (format matches world.py "CityA-CityB")
_CITY_ROUTES: dict[str, list[str]] = {
    "Mumbai":    ["Mumbai-Pune", "Mumbai-Ahmedabad", "Mumbai-Surat", "Delhi-Mumbai", "Chennai-Mumbai"],
    "Delhi":     ["Delhi-Jaipur", "Delhi-Ahmedabad", "Delhi-Kolkata", "Delhi-Mumbai"],
    "Kolkata":   ["Delhi-Kolkata", "Kolkata-Chennai", "Kolkata-Hyderabad"],
    "Chennai":   ["Kolkata-Chennai", "Chennai-Bangalore", "Chennai-Hyderabad", "Chennai-Mumbai"],
    "Bangalore": ["Chennai-Bangalore", "Bangalore-Hyderabad", "Bangalore-Pune"],
    "Hyderabad": ["Chennai-Hyderabad", "Bangalore-Hyderabad", "Hyderabad-Mumbai", "Pune-Hyderabad"],
    "Pune":      ["Mumbai-Pune", "Bangalore-Pune", "Pune-Hyderabad"],
    "Ahmedabad": ["Mumbai-Ahmedabad", "Delhi-Ahmedabad", "Ahmedabad-Surat", "Ahmedabad-Jaipur"],
    "Jaipur":    ["Delhi-Jaipur", "Ahmedabad-Jaipur"],
    "Surat":     ["Mumbai-Surat", "Ahmedabad-Surat"],
}


def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# ── 1. OpenWeatherMap ─────────────────────────────────────────────────────────

class WeatherConnector:
    """Polls OpenWeatherMap for each simulation city; maps severe weather to disruptions."""

    _OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")

    def fetch(self) -> list[dict[str, Any]]:
        if not self.api_key or not _REQUESTS_OK:
            return self._synthetic()
        disruptions: list[dict] = []
        for city in _CITIES:
            try:
                resp = requests.get(
                    self._OWM_URL,
                    params={"q": f"{city},IN", "appid": self.api_key, "units": "metric"},
                    timeout=5,
                )
                resp.raise_for_status()
                event = self._parse(city, resp.json())
                if event:
                    disruptions.append(event)
            except Exception:
                pass  # skip individual city failures; try next
        return disruptions if disruptions else self._synthetic()

    def _parse(self, city: str, data: dict) -> dict | None:
        w = (data.get("weather") or [{}])[0]
        weather_id  = w.get("id", 800)
        description = w.get("description", "clear sky")

        dtype: str | None = None
        severity = 0

        if 200 <= weather_id <= 232:    # thunderstorm
            dtype, severity = "flood", 4 if weather_id <= 202 else 3
        elif 300 <= weather_id <= 321:  # drizzle
            dtype, severity = "road_closure", 1
        elif 500 <= weather_id <= 504:  # rain
            dtype = "flood"
            severity = 3 if weather_id >= 502 else 2
        elif 511 <= weather_id <= 531:  # heavy / shower rain
            dtype, severity = "flood", 3
        elif weather_id == 781:         # tornado
            dtype, severity = "road_closure", 5
        elif 762 <= weather_id <= 780:  # volcanic ash, squall
            dtype, severity = "road_closure", 3
        elif 700 <= weather_id <= 761:  # fog, dust, haze
            dtype, severity = "road_closure", 2

        if severity < 2 or dtype is None:
            return None

        routes = _CITY_ROUTES.get(city, [])
        return {
            "type": dtype,
            "severity": severity,
            "affected_nodes": [city],
            "affected_routes": routes[:2],
            "turns_remaining": severity,
            "source": "openweathermap",
            "description": f"{description.title()} in {city}",
        }

    def _synthetic(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "flood",
                "severity": 3,
                "affected_nodes": ["Mumbai", "Pune"],
                "affected_routes": ["Mumbai-Pune"],
                "turns_remaining": 3,
                "source": "synthetic",
                "description": "Heavy monsoon rain — Mumbai–Pune corridor flooded (synthetic fallback)",
            },
            {
                "type": "road_closure",
                "severity": 2,
                "affected_nodes": ["Delhi"],
                "affected_routes": ["Delhi-Jaipur"],
                "turns_remaining": 2,
                "source": "synthetic",
                "description": "Dense fog causing road closures near Delhi (synthetic fallback)",
            },
        ]


# ── 2. ExchangeRate-API ───────────────────────────────────────────────────────

class ExchangeRateConnector:
    """
    Tracks USD/INR via open.er-api.com (no key needed).
    Sharp INR depreciation → tariff-pressure port_strike disruption on import hubs.
    """

    _URL = "https://open.er-api.com/v6/latest/USD"
    _INR_BASELINE = 83.0  # approximate long-run USD/INR midpoint

    def fetch(self) -> list[dict[str, Any]]:
        if not _REQUESTS_OK:
            return self._synthetic()
        try:
            resp = requests.get(self._URL, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            inr = data.get("rates", {}).get("INR", self._INR_BASELINE)
            return self._parse(inr)
        except Exception:
            return self._synthetic()

    def _parse(self, inr_rate: float) -> list[dict[str, Any]]:
        pct = ((inr_rate - self._INR_BASELINE) / self._INR_BASELINE) * 100
        if pct < 2.0:
            return []  # no meaningful shock

        severity = 2 if pct < 5 else (3 if pct < 10 else 4)
        return [{
            "type": "port_strike",
            "severity": severity,
            "affected_nodes": ["Mumbai", "Chennai"],
            "affected_routes": ["Mumbai-Pune", "Chennai-Bangalore"],
            "turns_remaining": severity + 1,
            "source": "exchangerate-api",
            "description": (
                f"INR at {inr_rate:.2f}/USD ({pct:+.1f}% vs baseline {self._INR_BASELINE}) — "
                f"import tariff pressure building on port cargo"
            ),
            "raw": {"inr_rate": round(inr_rate, 4), "depreciation_pct": round(pct, 2)},
        }]

    def _synthetic(self) -> list[dict[str, Any]]:
        return [{
            "type": "port_strike",
            "severity": 2,
            "affected_nodes": ["Mumbai", "Chennai"],
            "affected_routes": ["Mumbai-Pune", "Chennai-Bangalore"],
            "turns_remaining": 3,
            "source": "synthetic",
            "description": "INR ~3.5% below baseline — simulated tariff pressure on imports (synthetic fallback)",
            "raw": {"inr_rate": 86.0, "depreciation_pct": 3.6},
        }]


# ── 3. GDELT ──────────────────────────────────────────────────────────────────

class GDELTConnector:
    """
    Queries the GDELT 2.0 Doc API (no key needed) for India supply-chain
    disruption signals and maps high article-volume to road_closure / port_strike.
    """

    _URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    # (keyword, disruption_type, affected_nodes, affected_routes)
    _HOTSPOTS = [
        ("delhi",     "road_closure", ["Delhi", "Jaipur"],   ["Delhi-Jaipur", "Delhi-Ahmedabad"]),
        ("kolkata",   "port_strike",  ["Kolkata"],            ["Delhi-Kolkata", "Kolkata-Chennai"]),
        ("mumbai",    "port_strike",  ["Mumbai"],             ["Mumbai-Pune", "Mumbai-Ahmedabad"]),
        ("chennai",   "port_strike",  ["Chennai"],            ["Chennai-Bangalore", "Chennai-Hyderabad"]),
        ("rajasthan", "road_closure", ["Jaipur", "Ahmedabad"],["Delhi-Jaipur", "Ahmedabad-Jaipur"]),
    ]

    def fetch(self) -> list[dict[str, Any]]:
        if not _REQUESTS_OK:
            return self._synthetic()
        try:
            resp = requests.get(
                self._URL,
                params={
                    "query": "India supply chain disruption conflict strike protest border",
                    "mode": "artlist",
                    "maxrecords": "10",
                    "format": "json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception:
            return self._synthetic()

    def _parse(self, data: dict) -> list[dict[str, Any]]:
        articles = data.get("articles", [])
        if not articles:
            return self._synthetic()

        count = len(articles)
        severity = min(1 + count // 3, 4)
        combined_text = " ".join(
            (a.get("title", "") + " " + a.get("url", "")).lower()
            for a in articles
        )

        events: list[dict] = []
        for keyword, dtype, nodes, routes in self._HOTSPOTS:
            if keyword in combined_text:
                events.append({
                    "type": dtype,
                    "severity": severity,
                    "affected_nodes": nodes,
                    "affected_routes": routes,
                    "turns_remaining": severity,
                    "source": "gdelt",
                    "description": (
                        f"GDELT: {count} conflict/disruption articles mentioning "
                        f"'{keyword}' — elevated logistics risk"
                    ),
                })

        if not events:
            # Generic India-wide signal with capped severity
            events.append({
                "type": "road_closure",
                "severity": min(severity, 2),
                "affected_nodes": ["Delhi"],
                "affected_routes": ["Delhi-Jaipur"],
                "turns_remaining": 2,
                "source": "gdelt",
                "description": (
                    f"GDELT: {count} India disruption signals detected — "
                    f"applying conservative risk to Delhi corridor"
                ),
            })
        return events

    def _synthetic(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "road_closure",
                "severity": 2,
                "affected_nodes": ["Delhi", "Jaipur"],
                "affected_routes": ["Delhi-Jaipur"],
                "turns_remaining": 3,
                "source": "synthetic",
                "description": "Simulated: Rajasthan border tensions — Delhi-Jaipur corridor elevated risk (synthetic fallback)",
            },
            {
                "type": "port_strike",
                "severity": 3,
                "affected_nodes": ["Kolkata"],
                "affected_routes": ["Delhi-Kolkata", "Kolkata-Chennai"],
                "turns_remaining": 4,
                "source": "synthetic",
                "description": "Simulated: Dock-worker unrest at Kolkata port — clearance delays expected (synthetic fallback)",
            },
        ]


# ── Aggregator ────────────────────────────────────────────────────────────────

class LiveDataConnector:
    """Aggregates all three live sources into a single disruption payload."""

    def __init__(self) -> None:
        self.weather  = WeatherConnector()
        self.exchange = ExchangeRateConnector()
        self.gdelt    = GDELTConnector()

    def get_weather_disruptions(self) -> list[dict[str, Any]]:
        return self.weather.fetch()

    def get_exchange_shocks(self) -> list[dict[str, Any]]:
        return self.exchange.fetch()

    def get_geopolitical_zones(self) -> list[dict[str, Any]]:
        return self.gdelt.fetch()

    def get_all_disruptions(self) -> dict[str, Any]:
        weather      = self.get_weather_disruptions()
        currency     = self.get_exchange_shocks()
        geopolitical = self.get_geopolitical_zones()
        all_events   = weather + currency + geopolitical
        return {
            "timestamp":         _utc_now(),
            "weather":           weather,
            "currency":          currency,
            "geopolitical":      geopolitical,
            "total_disruptions": len(all_events),
            "max_severity":      max((e["severity"] for e in all_events), default=0),
        }
