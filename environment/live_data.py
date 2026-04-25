"""
Live data connectors for LogiCrisis.
All connectors fail silently — never block env.reset().

Sources (zero-config unless noted):
  1. Open-Meteo        — free, no key, real India weather
  2. OpenWeatherMap    — free key (OPENWEATHERMAP_API_KEY), more detail
  3. ExchangeRate-API  — free, no key, live USD/INR
  4. GDELT 2.0         — free, no key, India conflict/strike signals
  5. World Bank        — free, no key, crude oil price (fuel cost proxy)
  6. NewsAPI           — free key (NEWS_API_KEY), India trade/port news

Each connector returns strongly-typed results with human-readable descriptions
so agents can read them directly in observation text.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

from .models import Disruption, DisruptionType

# ── City topology (matches world.py) ─────────────────────────────────────────

_CITIES = [
    "Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat",
]

# Open-Meteo / lat-lon for each city
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Kolkata":   (22.5726, 88.3639),
    "Chennai":   (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur":    (26.9124, 75.7873),
    "Surat":     (21.1702, 72.8311),
}

_CITY_ROUTES: dict[str, list[str]] = {
    "Mumbai":    ["Mumbai-Pune", "Mumbai-Ahmedabad", "Mumbai-Surat"],
    "Delhi":     ["Delhi-Jaipur", "Delhi-Ahmedabad"],
    "Kolkata":   ["Kolkata-Hyderabad"],
    "Chennai":   ["Chennai-Bangalore", "Chennai-Hyderabad"],
    "Bangalore": ["Chennai-Bangalore", "Bangalore-Hyderabad", "Bangalore-Pune"],
    "Hyderabad": ["Chennai-Hyderabad", "Bangalore-Hyderabad", "Pune-Hyderabad"],
    "Pune":      ["Mumbai-Pune", "Bangalore-Pune", "Pune-Hyderabad"],
    "Ahmedabad": ["Mumbai-Ahmedabad", "Delhi-Ahmedabad", "Ahmedabad-Surat", "Ahmedabad-Jaipur"],
    "Jaipur":    ["Delhi-Jaipur", "Ahmedabad-Jaipur"],
    "Surat":     ["Mumbai-Surat", "Ahmedabad-Surat"],
}

# Regions for each city
_CITY_REGION: dict[str, str] = {
    "Mumbai": "West", "Pune": "West", "Surat": "West", "Ahmedabad": "West",
    "Delhi": "North", "Jaipur": "North",
    "Kolkata": "East",
    "Chennai": "South", "Bangalore": "South", "Hyderabad": "South",
}


# ── Typed result objects ──────────────────────────────────────────────────────

@dataclass
class WeatherAlert:
    city: str
    region: str
    condition: str          # e.g. "Heavy Rain", "Thunderstorm"
    severity: int           # 1-5
    temp_celsius: float
    humidity_pct: int
    wind_kmh: float
    disrupts_routes: list[str]
    source: str             # "open_meteo" or "openweathermap"
    description: str        # human-readable for agent prompt

    def to_disruption(self) -> Disruption:
        dtype = DisruptionType.FLOOD if "rain" in self.condition.lower() else DisruptionType.ROAD_CLOSURE
        return Disruption(
            disruption_type=dtype,
            severity=self.severity,
            affected_nodes=[self.city],
            affected_routes=self.disrupts_routes[:2],
            turns_remaining=self.severity,
        )


@dataclass
class CurrencySignal:
    pair: str               # e.g. "USD/INR"
    rate: float
    baseline: float
    swing_pct: float
    severity: int           # 1-5
    affected_ports: list[str]
    description: str        # human-readable for agent prompt
    shock_active: bool

    def to_disruption(self) -> Optional[Disruption]:
        if not self.shock_active:
            return None
        return Disruption(
            disruption_type=DisruptionType.PORT_STRIKE,
            severity=self.severity,
            affected_nodes=self.affected_ports,
            affected_routes=[],
            turns_remaining=self.severity + 1,
        )


@dataclass
class ConflictSignal:
    source: str             # "gdelt" or "newsapi"
    affected_cities: list[str]
    severity: int           # 1-5
    keywords_found: list[str]
    description: str        # human-readable for agent prompt

    def to_disruption(self) -> Optional[Disruption]:
        if not self.affected_cities:
            return None
        return Disruption(
            disruption_type=DisruptionType.ROAD_CLOSURE,
            severity=self.severity,
            affected_nodes=self.affected_cities,
            affected_routes=[],
            turns_remaining=3,
        )


@dataclass
class CommoditySignal:
    commodity: str          # e.g. "Crude Oil"
    price_usd: float
    baseline_usd: float
    change_pct: float
    impact: str             # "transport_cost_up" / "transport_cost_down"
    description: str        # human-readable for agent prompt


@dataclass
class LiveContext:
    """
    Aggregated API data passed directly into agent observation text.
    Agents read this and reason about it in their actions.
    """
    weather_alerts: list[WeatherAlert] = field(default_factory=list)
    currency_signal: Optional[CurrencySignal] = None
    conflict_signal: Optional[ConflictSignal] = None
    commodity_signal: Optional[CommoditySignal] = None
    fetch_timestamp: str = ""

    def to_prompt_lines(self) -> list[str]:
        """Convert live data into human-readable lines for agent observation."""
        lines = []
        if self.weather_alerts:
            lines.append(f"LIVE WEATHER ALERTS ({len(self.weather_alerts)} cities affected):")
            for a in self.weather_alerts[:4]:   # cap to 4 to avoid prompt bloat
                lines.append(f"  - {a.city} ({a.region}): {a.description}")

        if self.currency_signal and self.currency_signal.shock_active:
            lines.append(f"LIVE CURRENCY SIGNAL: {self.currency_signal.description}")

        if self.conflict_signal and self.conflict_signal.affected_cities:
            lines.append(f"LIVE CONFLICT SIGNAL: {self.conflict_signal.description}")

        if self.commodity_signal:
            lines.append(f"LIVE COMMODITY SIGNAL: {self.commodity_signal.description}")

        return lines

    def is_empty(self) -> bool:
        return (not self.weather_alerts
                and (self.currency_signal is None or not self.currency_signal.shock_active)
                and (self.conflict_signal is None or not self.conflict_signal.affected_cities)
                and self.commodity_signal is None)


# ── 1. Open-Meteo (NO KEY REQUIRED) ──────────────────────────────────────────

class OpenMeteoConnector:
    """
    Free weather API — no key, no rate limit.
    Returns real weather for all 10 Indian cities.
    Docs: https://open-meteo.com/en/docs
    """

    _URL = "https://api.open-meteo.com/v1/forecast"

    # WMO weather code → (condition_name, severity, is_disruption)
    _WMO_CODES = {
        # Thunderstorms
        95: ("Thunderstorm", 4, True),
        96: ("Hail Storm", 4, True),
        99: ("Heavy Thunderstorm with Hail", 5, True),
        # Heavy rain
        65: ("Heavy Rain", 3, True),
        67: ("Heavy Freezing Rain", 4, True),
        63: ("Moderate Rain", 2, True),
        61: ("Light Rain", 1, False),
        # Snow (rare in India but possible in North)
        75: ("Heavy Snow", 4, True),
        73: ("Moderate Snow", 3, True),
        # Fog
        45: ("Dense Fog", 2, True),
        48: ("Thick Fog", 3, True),
        # Drizzle
        53: ("Moderate Drizzle", 1, False),
        55: ("Heavy Drizzle", 2, True),
    }

    def fetch(self) -> list[WeatherAlert]:
        if not _REQUESTS_OK:
            return []
        alerts = []
        for city, (lat, lon) in _CITY_COORDS.items():
            try:
                resp = requests.get(
                    self._URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                        "wind_speed_unit": "kmh",
                        "timezone": "Asia/Kolkata",
                        "forecast_days": 1,
                    },
                    timeout=6,
                )
                resp.raise_for_status()
                alert = self._parse(city, resp.json())
                if alert:
                    alerts.append(alert)
            except Exception:
                pass
        return alerts

    def _parse(self, city: str, data: dict) -> Optional[WeatherAlert]:
        current = data.get("current", {})
        code = current.get("weather_code", 0)
        condition, severity, is_disruption = self._WMO_CODES.get(
            code, ("Clear", 0, False)
        )
        if not is_disruption:
            return None

        temp = current.get("temperature_2m", 25.0)
        humidity = int(current.get("relative_humidity_2m", 60))
        wind = current.get("wind_speed_10m", 0.0)
        routes = _CITY_ROUTES.get(city, [])
        region = _CITY_REGION.get(city, "Unknown")

        desc = (
            f"{condition} (severity={severity}) — temp {temp:.0f}C, "
            f"wind {wind:.0f}km/h. Routes at risk: {routes[:2]}"
        )
        return WeatherAlert(
            city=city, region=region,
            condition=condition, severity=severity,
            temp_celsius=temp, humidity_pct=humidity, wind_kmh=wind,
            disrupts_routes=routes, source="open_meteo",
            description=desc,
        )


# ── 2. OpenWeatherMap (FREE KEY — optional, richer data) ─────────────────────

class OpenWeatherMapConnector:
    """
    Richer weather data when OPENWEATHERMAP_API_KEY is set.
    Free tier: 1,000,000 calls/month, 60/min.
    Get key at: https://openweathermap.org/api → One Call API
    """

    _URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self) -> None:
        self.api_key = os.environ.get("scaler_weather", "")

    def is_available(self) -> bool:
        return bool(self.api_key) and _REQUESTS_OK

    def fetch(self) -> list[WeatherAlert]:
        if not self.is_available():
            return []
        alerts = []
        for city in _CITIES:
            try:
                resp = requests.get(
                    self._URL,
                    params={"q": f"{city},IN", "appid": self.api_key, "units": "metric"},
                    timeout=5,
                )
                resp.raise_for_status()
                alert = self._parse(city, resp.json())
                if alert:
                    alerts.append(alert)
            except Exception:
                pass
        return alerts

    def _parse(self, city: str, data: dict) -> Optional[WeatherAlert]:
        w = (data.get("weather") or [{}])[0]
        wid = w.get("id", 800)
        desc_raw = w.get("description", "clear")
        main = data.get("main", {})
        wind = data.get("wind", {})
        temp = main.get("temp", 25.0)
        humidity = main.get("humidity", 60)
        wind_speed = wind.get("speed", 0) * 3.6   # m/s → km/h

        # Map OWM codes to severity
        if 200 <= wid <= 232:
            condition, severity, is_dis = "Thunderstorm", 4, True
        elif 502 <= wid <= 531:
            condition, severity, is_dis = "Heavy Rain", 3, True
        elif 500 <= wid <= 501:
            condition, severity, is_dis = "Moderate Rain", 2, True
        elif wid == 781:
            condition, severity, is_dis = "Tornado", 5, True
        elif 700 <= wid <= 780:
            condition, severity, is_dis = "Dense Fog/Haze", 2, True
        else:
            return None

        routes = _CITY_ROUTES.get(city, [])
        region = _CITY_REGION.get(city, "Unknown")
        return WeatherAlert(
            city=city, region=region,
            condition=condition, severity=severity,
            temp_celsius=temp, humidity_pct=humidity, wind_kmh=wind_speed,
            disrupts_routes=routes, source="openweathermap",
            description=(
                f"{condition} ({desc_raw}, severity={severity}) — "
                f"temp {temp:.0f}C. Routes at risk: {routes[:2]}"
            ),
        )


# ── 3. ExchangeRate-API (NO KEY REQUIRED) ────────────────────────────────────

class ExchangeRateConnector:
    """
    Live USD/INR from open.er-api.com.
    A swing >5% vs baseline = tariff shock.
    Free, no key, no rate limit.
    Docs: https://www.exchangerate-api.com/docs/free
    """

    _URL = "https://open.er-api.com/v6/latest/USD"
    _INR_BASELINE = 83.5

    def fetch(self) -> CurrencySignal:
        if not _REQUESTS_OK:
            return self._neutral()
        try:
            resp = requests.get(self._URL, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            inr = data.get("rates", {}).get("INR", self._INR_BASELINE)
            # Also fetch EUR/INR for cross-currency signal
            eur = data.get("rates", {}).get("EUR", 0.9)
            return self._parse(inr, eur)
        except Exception:
            return self._neutral()

    def _parse(self, inr: float, eur: float) -> CurrencySignal:
        swing = ((inr - self._INR_BASELINE) / self._INR_BASELINE) * 100
        shock = abs(swing) > 5.0
        severity = 0 if not shock else (2 if abs(swing) < 8 else (3 if abs(swing) < 12 else 4))

        if shock:
            direction = "depreciation" if swing > 0 else "appreciation"
            desc = (
                f"INR at {inr:.2f}/USD ({swing:+.1f}% vs baseline {self._INR_BASELINE}). "
                f"INR {direction} → import tariff pressure on port cargo. "
                f"Severity={severity}. Customs Broker: negotiate bypass NOW."
            )
        else:
            desc = f"INR at {inr:.2f}/USD ({swing:+.1f}% vs baseline) — stable, no tariff shock."

        return CurrencySignal(
            pair="USD/INR",
            rate=inr,
            baseline=self._INR_BASELINE,
            swing_pct=round(swing, 2),
            severity=severity,
            affected_ports=["Mumbai", "Chennai", "Kolkata"],
            description=desc,
            shock_active=shock,
        )

    def _neutral(self) -> CurrencySignal:
        return CurrencySignal(
            pair="USD/INR", rate=self._INR_BASELINE, baseline=self._INR_BASELINE,
            swing_pct=0.0, severity=0, affected_ports=[],
            description="Exchange rate data unavailable.", shock_active=False,
        )


# ── 4. GDELT 2.0 (NO KEY REQUIRED) ───────────────────────────────────────────

class GDELTConnector:
    """
    GDELT 2.0 Doc API — scans global news for India supply chain signals.
    Free, no key, unlimited.
    Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
    """

    _URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Keywords that indicate route-blocking events
    _DISRUPTION_KEYWORDS = [
        "strike", "protest", "blockade", "riot", "clash",
        "flood", "landslide", "accident", "closure",
    ]
    _SEVERITY_KEYWORDS = {
        5: ["war", "military", "armed conflict"],
        4: ["riot", "violence", "blockade"],
        3: ["strike", "protest", "closure"],
        2: ["flood", "accident", "landslide"],
    }

    def fetch(self) -> ConflictSignal:
        if not _REQUESTS_OK:
            return self._empty()
        try:
            resp = requests.get(
                self._URL,
                params={
                    "query": (
                        "India supply chain port strike protest road closure "
                        "flood cargo disruption logistics"
                    ),
                    "mode": "artlist",
                    "maxrecords": "15",
                    "format": "json",
                    "timespan": "1d",     # last 24 hours only
                },
                timeout=12,
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception:
            return self._empty()

    def _parse(self, data: dict) -> ConflictSignal:
        articles = data.get("articles", [])
        if not articles:
            return self._empty()

        combined = " ".join(
            (a.get("title", "") + " " + a.get("url", "")).lower()
            for a in articles
        )

        affected = [c for c in _CITIES if c.lower() in combined]
        found_keywords = [kw for kw in self._DISRUPTION_KEYWORDS if kw in combined]

        severity = 1
        for sev, keywords in sorted(self._SEVERITY_KEYWORDS.items(), reverse=True):
            if any(kw in combined for kw in keywords):
                severity = sev
                break

        if not affected and not found_keywords:
            return self._empty()

        desc = (
            f"GDELT: {len(articles)} India logistics articles in last 24h. "
            f"Keywords: {found_keywords[:4]}. "
            f"Cities mentioned: {affected[:4]}. "
            f"Severity={severity}. GeoAnalyst: issue alert for affected corridors."
        )
        return ConflictSignal(
            source="gdelt",
            affected_cities=affected[:4],
            severity=severity,
            keywords_found=found_keywords[:5],
            description=desc,
        )

    def _empty(self) -> ConflictSignal:
        return ConflictSignal(
            source="gdelt", affected_cities=[], severity=0,
            keywords_found=[], description="No GDELT conflict signals detected.",
        )


# ── 5. NewsAPI (FREE KEY — optional, breaking news) ──────────────────────────

class NewsAPIConnector:
    """
    Real-time India trade/port/logistics headlines.
    Free tier: 100 requests/day.
    Get key at: https://newsapi.org/register → copy API key → set NEWS_API_KEY
    """

    _URL = "https://newsapi.org/v2/everything"

    def __init__(self) -> None:
        self.api_key = os.environ.get("NEWS_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key) and _REQUESTS_OK

    def fetch(self) -> ConflictSignal:
        if not self.is_available():
            return ConflictSignal(
                source="newsapi", affected_cities=[], severity=0,
                keywords_found=[], description="NewsAPI key not set (set NEWS_API_KEY).",
            )
        try:
            resp = requests.get(
                self._URL,
                params={
                    "q": "India port strike logistics supply chain disruption",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10,
                    "apiKey": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception:
            return ConflictSignal(
                source="newsapi", affected_cities=[], severity=0,
                keywords_found=[], description="NewsAPI fetch failed.",
            )

    def _parse(self, data: dict) -> ConflictSignal:
        articles = data.get("articles", [])
        if not articles:
            return ConflictSignal(
                source="newsapi", affected_cities=[], severity=0,
                keywords_found=[], description="No breaking India trade news.",
            )

        combined = " ".join(
            ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
            for a in articles
        )
        affected = [c for c in _CITIES if c.lower() in combined]
        keywords = ["strike", "port", "protest", "closure", "flood", "blockade"]
        found = [kw for kw in keywords if kw in combined]
        severity = min(len(found) + 1, 5)

        headlines = [a.get("title", "")[:80] for a in articles[:3]]
        desc = (
            f"NewsAPI: {len(articles)} India trade articles. "
            f"Top: \"{headlines[0]}\". "
            f"Cities: {affected[:3]}. Severity={severity}."
        )
        return ConflictSignal(
            source="newsapi",
            affected_cities=affected[:4],
            severity=severity,
            keywords_found=found,
            description=desc,
        )


# ── 6. World Bank Commodity Prices (NO KEY REQUIRED) ─────────────────────────

class WorldBankCommodityConnector:
    """
    Live commodity prices from World Bank API.
    Crude oil price → transport cost signal for Carriers and Insurers.
    Free, no key.
    Docs: https://api.worldbank.org/v2
    """

    # Crude oil indicator: PLOGORE (Crude oil, avg spot price)
    _URL = "https://api.worldbank.org/v2/en/indicator/PLOGORE"
    _BASELINE_USD_BBL = 80.0   # ~2024 average

    def fetch(self) -> Optional[CommoditySignal]:
        if not _REQUESTS_OK:
            return None
        try:
            resp = requests.get(
                self._URL,
                params={"format": "json", "mrv": 2, "per_page": 2},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            # World Bank returns [metadata, [records...]]
            records = data[1] if isinstance(data, list) and len(data) > 1 else []
            if not records:
                return None
            latest = records[0]
            price = latest.get("value")
            if price is None:
                return None
            return self._parse(float(price))
        except Exception:
            return None

    def _parse(self, price: float) -> CommoditySignal:
        change = ((price - self._BASELINE_USD_BBL) / self._BASELINE_USD_BBL) * 100
        if change > 10:
            impact = "transport_cost_up"
            desc = (
                f"Crude oil at ${price:.1f}/bbl ({change:+.1f}% vs baseline). "
                f"Fuel costs rising — Carriers: factor surcharge into bid prices. "
                f"Insurers: raise cargo insurance premiums by ~{change:.0f}%."
            )
        elif change < -10:
            impact = "transport_cost_down"
            desc = (
                f"Crude oil at ${price:.1f}/bbl ({change:+.1f}% vs baseline). "
                f"Fuel costs falling — Carriers: competitive pricing window. "
                f"Pass savings to Shippers via lower bid prices."
            )
        else:
            impact = "stable"
            desc = f"Crude oil at ${price:.1f}/bbl ({change:+.1f}% vs baseline) — stable fuel costs."

        return CommoditySignal(
            commodity="Crude Oil",
            price_usd=price,
            baseline_usd=self._BASELINE_USD_BBL,
            change_pct=round(change, 1),
            impact=impact,
            description=desc,
        )


# ── Master aggregator ─────────────────────────────────────────────────────────

class LiveDataConnector:
    """
    Aggregates all live data sources into a LiveContext object.
    Priority: Open-Meteo (no key) + OWM (if key set) → deduplicate cities.
    """

    def __init__(self) -> None:
        self.open_meteo  = OpenMeteoConnector()
        self.owm         = OpenWeatherMapConnector()
        self.exchange    = ExchangeRateConnector()
        self.gdelt       = GDELTConnector()
        self.newsapi     = NewsAPIConnector()
        self.commodity   = WorldBankCommodityConnector()

    def get_live_context(self) -> LiveContext:
        """Fetch all sources and return a unified LiveContext."""
        # Weather: prefer OWM detail if available, fall back to Open-Meteo
        if self.owm.is_available():
            weather = self.owm.fetch()
        else:
            weather = self.open_meteo.fetch()

        # Deduplicate cities (keep highest severity per city)
        best: dict[str, WeatherAlert] = {}
        for a in weather:
            if a.city not in best or a.severity > best[a.city].severity:
                best[a.city] = a
        weather = list(best.values())

        # Currency
        currency = self.exchange.fetch()

        # Conflict: merge GDELT + NewsAPI (take whichever has more affected cities)
        gdelt_sig = self.gdelt.fetch()
        news_sig  = self.newsapi.fetch() if self.newsapi.is_available() else None

        if news_sig and len(news_sig.affected_cities) > len(gdelt_sig.affected_cities):
            conflict = news_sig
        else:
            conflict = gdelt_sig

        # Commodity
        commodity = self.commodity.fetch()

        return LiveContext(
            weather_alerts=weather,
            currency_signal=currency,
            conflict_signal=conflict,
            commodity_signal=commodity,
            fetch_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # ── Legacy methods (keep backward compat with existing env.py calls) ───────

    def get_weather_disruptions(self) -> list[Disruption]:
        alerts = (self.owm.fetch() if self.owm.is_available()
                  else self.open_meteo.fetch())
        return [a.to_disruption() for a in alerts]

    def get_exchange_shocks(self):
        sig = self.exchange.fetch()
        if not sig.shock_active:
            return None
        # Return a legacy-compatible object with .severity and .affected_cities
        return sig

    def get_geopolitical_zones(self) -> list[str]:
        return self.gdelt.fetch().affected_cities

    def get_all_disruptions(self) -> dict:
        ctx = self.get_live_context()
        return {
            "weather_alerts": [
                {
                    "city": a.city, "condition": a.condition, "severity": a.severity,
                    "routes_at_risk": a.disrupts_routes, "source": a.source,
                }
                for a in ctx.weather_alerts
            ],
            "currency": {
                "pair": ctx.currency_signal.pair,
                "rate": ctx.currency_signal.rate,
                "swing_pct": ctx.currency_signal.swing_pct,
                "shock_active": ctx.currency_signal.shock_active,
                "severity": ctx.currency_signal.severity,
            } if ctx.currency_signal else None,
            "conflict": {
                "source": ctx.conflict_signal.source,
                "affected_cities": ctx.conflict_signal.affected_cities,
                "severity": ctx.conflict_signal.severity,
                "keywords": ctx.conflict_signal.keywords_found,
            } if ctx.conflict_signal else None,
            "commodity": {
                "name": ctx.commodity_signal.commodity,
                "price_usd": ctx.commodity_signal.price_usd,
                "change_pct": ctx.commodity_signal.change_pct,
                "impact": ctx.commodity_signal.impact,
            } if ctx.commodity_signal else None,
            "timestamp": ctx.fetch_timestamp,
        }
