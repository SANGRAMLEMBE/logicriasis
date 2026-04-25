"""
Live data connectors for LogiCrisis.
  - OpenWeatherMap  (free, set OPENWEATHERMAP_API_KEY) → list[Disruption]
  - ExchangeRate-API (free, no key: api.exchangerate-api.com) → GeopoliticalEvent | None
  - GDELT 2.0 Doc   (free, no key: api.gdeltproject.org) → list[str] (city names)

All connectors wrapped in try/except — any failure returns empty list / None silently.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

from .models import Disruption, DisruptionType

# ── City catalogue (matches world.py topology) ────────────────────────────────

_CITIES: list[str] = [
    "Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat",
]

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


# ── GeopoliticalEvent (returned by ExchangeRateConnector) ────────────────────

@dataclass
class GeopoliticalEvent:
    event_type: str               # e.g. "tariff_shock"
    severity: int                 # 1–5
    affected_cities: list[str]    # simulation city names impacted
    description: str
    currency_pair: str = ""       # e.g. "USD/INR"
    swing_pct: float = 0.0        # percentage change from baseline


# ── 1. OpenWeatherMap → list[Disruption] ─────────────────────────────────────

class WeatherConnector:
    """Polls OpenWeatherMap for each simulation city; maps storms/cyclones to Disruption objects."""

    _URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")

    def fetch(self) -> list[Disruption]:
        if not self.api_key or not _REQUESTS_OK:
            return []
        disruptions: list[Disruption] = []
        for city in _CITIES:
            try:
                resp = requests.get(
                    self._URL,
                    params={"q": f"{city},IN", "appid": self.api_key, "units": "metric"},
                    timeout=5,
                )
                resp.raise_for_status()
                d = self._parse(city, resp.json())
                if d:
                    disruptions.append(d)
            except Exception:
                pass
        return disruptions

    def _parse(self, city: str, data: dict) -> Optional[Disruption]:
        w = (data.get("weather") or [{}])[0]
        weather_id = w.get("id", 800)

        dtype: Optional[DisruptionType] = None
        severity = 0

        if 200 <= weather_id <= 232:    # thunderstorm / cyclone
            dtype, severity = DisruptionType.FLOOD, 4 if weather_id <= 202 else 3
        elif 502 <= weather_id <= 531:  # heavy / violent rain
            dtype, severity = DisruptionType.FLOOD, 3
        elif 500 <= weather_id <= 501:  # moderate rain
            dtype, severity = DisruptionType.FLOOD, 2
        elif weather_id == 781:         # tornado
            dtype, severity = DisruptionType.ROAD_CLOSURE, 5
        elif 762 <= weather_id <= 780:  # volcanic ash / squall
            dtype, severity = DisruptionType.ROAD_CLOSURE, 3
        elif 700 <= weather_id <= 761:  # fog / dust / haze
            dtype, severity = DisruptionType.ROAD_CLOSURE, 2

        if severity < 2 or dtype is None:
            return None

        routes = _CITY_ROUTES.get(city, [])
        return Disruption(
            disruption_type=dtype,
            severity=severity,
            affected_nodes=[city],
            affected_routes=routes[:2],
            turns_remaining=severity,
        )


# ── 2. ExchangeRate-API → GeopoliticalEvent | None ───────────────────────────

class ExchangeRateConnector:
    """
    Fetches live USD/INR from api.exchangerate-api.com (no key needed).
    A swing > 5% vs baseline is treated as a tariff shock GeopoliticalEvent.
    """

    _URL = "https://open.er-api.com/v6/latest/USD"
    _INR_BASELINE = 83.0

    def fetch(self) -> Optional[GeopoliticalEvent]:
        if not _REQUESTS_OK:
            return None
        try:
            resp = requests.get(self._URL, timeout=8)
            resp.raise_for_status()
            inr = resp.json().get("rates", {}).get("INR", self._INR_BASELINE)
            return self._parse(inr)
        except Exception:
            return None

    def _parse(self, inr_rate: float) -> Optional[GeopoliticalEvent]:
        swing = ((inr_rate - self._INR_BASELINE) / self._INR_BASELINE) * 100
        if abs(swing) <= 5.0:
            return None

        severity = 2 if abs(swing) < 8 else (3 if abs(swing) < 12 else 4)
        return GeopoliticalEvent(
            event_type="tariff_shock",
            severity=severity,
            affected_cities=["Mumbai", "Chennai"],   # major import ports
            description=(
                f"INR at {inr_rate:.2f}/USD ({swing:+.1f}% vs baseline) — "
                f"import tariff pressure building on port cargo"
            ),
            currency_pair="USD/INR",
            swing_pct=round(swing, 2),
        )


# ── 3. GDELT → list[str] (affected city names) ───────────────────────────────

class GDELTConnector:
    """
    Queries GDELT 2.0 Doc API for India supply-chain conflict signals.
    Returns list of simulation city names mentioned in recent disruption articles.
    """

    _URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(self) -> list[str]:
        if not _REQUESTS_OK:
            return []
        try:
            resp = requests.get(
                self._URL,
                params={
                    "query": "India supply chain disruption conflict strike protest port",
                    "mode": "artlist",
                    "maxrecords": "10",
                    "format": "json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception:
            return []

    def _parse(self, data: dict) -> list[str]:
        articles = data.get("articles", [])
        if not articles:
            return []
        combined = " ".join(
            a.get("title", "") + " " + a.get("url", "")
            for a in articles
        ).lower()
        return [city for city in _CITIES if city.lower() in combined]


# ── Aggregator ────────────────────────────────────────────────────────────────

class LiveDataConnector:
    """Aggregates all three live sources into a unified disruption payload."""

    def __init__(self) -> None:
        self.weather  = WeatherConnector()
        self.exchange = ExchangeRateConnector()
        self.gdelt    = GDELTConnector()

    def get_weather_disruptions(self) -> list[Disruption]:
        return self.weather.fetch()

    def get_exchange_shocks(self) -> Optional[GeopoliticalEvent]:
        return self.exchange.fetch()

    def get_geopolitical_zones(self) -> list[str]:
        return self.gdelt.fetch()

    def get_all_disruptions(self) -> dict:
        weather      = self.get_weather_disruptions()
        shock        = self.get_exchange_shocks()
        geo_cities   = self.get_geopolitical_zones()
        return {
            "weather_disruptions": [
                {
                    "type": d.disruption_type.value,
                    "severity": d.severity,
                    "affected_nodes": d.affected_nodes,
                    "affected_routes": d.affected_routes,
                    "turns_remaining": d.turns_remaining,
                }
                for d in weather
            ],
            "currency_shock": (
                {
                    "event_type": shock.event_type,
                    "severity": shock.severity,
                    "affected_cities": shock.affected_cities,
                    "description": shock.description,
                    "currency_pair": shock.currency_pair,
                    "swing_pct": shock.swing_pct,
                }
                if shock else None
            ),
            "geopolitical_cities": geo_cities,
            "total_weather_events": len(weather),
            "tariff_shock_active": shock is not None,
            "conflict_cities": geo_cities,
        }
