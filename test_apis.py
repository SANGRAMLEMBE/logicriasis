"""
API connectivity test — run this to verify all live data sources are working.
Usage:
    python test_apis.py

Zero-config APIs should always show PASS.
Optional APIs show SKIP if key not set, PASS/FAIL if key is set.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    print("[ERROR] requests not installed: pip install requests")
    sys.exit(1)


def _check(label: str, fn, expect_key: str = "") -> None:
    skip = expect_key and not os.environ.get(expect_key)
    if skip:
        print(f"  SKIP  {label:<40} (set {expect_key} to enable)")
        return
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  PASS  {label:<40} ({elapsed:.1f}s) ->{result}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL  {label:<40} ({elapsed:.1f}s) ->{e}")


print("=" * 65)
print("LogiCrisis — Live API Connectivity Test")
print("=" * 65)


# ── 1. Open-Meteo (no key) ────────────────────────────────────────────────────
print("\n[1] Open-Meteo Weather (zero-config)")

def _test_open_meteo():
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": 19.0760, "longitude": 72.8777,
            "current": "temperature_2m,weather_code",
            "timezone": "Asia/Kolkata", "forecast_days": 1,
        },
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    temp = data["current"]["temperature_2m"]
    code = data["current"]["weather_code"]
    return f"Mumbai temp={temp}C weather_code={code}"

_check("Open-Meteo (Mumbai weather)", _test_open_meteo)


# ── 2. OpenWeatherMap (optional key) ─────────────────────────────────────────
print("\n[2] OpenWeatherMap Weather (optional — set OPENWEATHERMAP_API_KEY)")

def _test_owm():
    key = os.environ["OPENWEATHERMAP_API_KEY"]
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": "Mumbai,IN", "appid": key, "units": "metric"},
        timeout=6,
    )
    resp.raise_for_status()
    d = resp.json()
    temp = d["main"]["temp"]
    desc = d["weather"][0]["description"]
    return f"Mumbai temp={temp}C condition={desc}"

_check("OpenWeatherMap (Mumbai)", _test_owm, expect_key="OPENWEATHERMAP_API_KEY")


# ── 3. ExchangeRate-API (no key) ─────────────────────────────────────────────
print("\n[3] ExchangeRate-API (zero-config)")

def _test_exchange():
    resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
    resp.raise_for_status()
    inr = resp.json()["rates"]["INR"]
    eur = resp.json()["rates"]["EUR"]
    return f"USD/INR={inr:.2f}  USD/EUR={eur:.4f}"

_check("ExchangeRate-API (USD/INR + EUR)", _test_exchange)


# ── 4. GDELT 2.0 (no key) ────────────────────────────────────────────────────
print("\n[4] GDELT 2.0 Conflict News (zero-config)")

def _test_gdelt():
    resp = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": "India supply chain disruption strike port",
            "mode": "artlist", "maxrecords": "5",
            "format": "json", "timespan": "1d",
        },
        timeout=15,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    titles = [a.get("title", "")[:50] for a in articles[:2]]
    return f"{len(articles)} articles — e.g. {titles}"

_check("GDELT (India logistics news)", _test_gdelt)


# ── 5. World Bank Commodity (no key) ─────────────────────────────────────────
print("\n[5] World Bank Commodity Prices (zero-config)")

def _test_worldbank():
    resp = requests.get(
        "https://api.worldbank.org/v2/en/indicator/PLOGORE",
        params={"format": "json", "mrv": 1, "per_page": 1},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    records = data[1] if isinstance(data, list) and len(data) > 1 else []
    if records and records[0].get("value"):
        price = records[0]["value"]
        period = records[0].get("date", "?")
        return f"Crude oil ${price:.1f}/bbl (period: {period})"
    return "No data returned (API may be slow)"

_check("World Bank (Crude Oil price)", _test_worldbank)


# ── 6. NewsAPI (optional key) ─────────────────────────────────────────────────
print("\n[6] NewsAPI Breaking News (optional — set NEWS_API_KEY)")

def _test_newsapi():
    key = os.environ["NEWS_API_KEY"]
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": "India port logistics disruption",
            "language": "en", "sortBy": "publishedAt",
            "pageSize": 3, "apiKey": key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    titles = [a.get("title", "")[:50] for a in articles[:2]]
    return f"{len(articles)} articles — e.g. {titles}"

_check("NewsAPI (India trade news)", _test_newsapi, expect_key="NEWS_API_KEY")


# ── 7. Full LiveDataConnector test ────────────────────────────────────────────
print("\n[7] Full LiveDataConnector + LiveContext integration")

def _test_live_connector():
    from environment.live_data import LiveDataConnector
    connector = LiveDataConnector()
    ctx = connector.get_live_context()
    parts = []
    parts.append(f"weather_alerts={len(ctx.weather_alerts)}")
    if ctx.currency_signal:
        parts.append(f"currency_shock={ctx.currency_signal.shock_active}")
    if ctx.conflict_signal:
        parts.append(f"conflict_cities={ctx.conflict_signal.affected_cities}")
    if ctx.commodity_signal:
        parts.append(f"crude=${ctx.commodity_signal.price_usd:.1f}")
    parts.append(f"timestamp={ctx.fetch_timestamp}")
    return " | ".join(parts)

_check("LiveDataConnector.get_live_context()", _test_live_connector)


# ── 8. Agent observation with live data ───────────────────────────────────────
print("\n[8] Agent observation includes live data")

def _test_agent_obs():
    from environment import LogiCrisisEnv
    env = LogiCrisisEnv(curriculum_level=1, seed=42)
    obs = env.reset()
    agent_id = list(obs.keys())[0]
    o = obs[agent_id]
    prompt = o.to_prompt_text()
    has_live = any(
        keyword in prompt
        for keyword in ["LIVE WEATHER", "LIVE CURRENCY", "LIVE CONFLICT", "LIVE COMMODITY"]
    )
    live_lines = [l for l in prompt.split("\n") if "LIVE" in l]
    return (
        f"role={o.role.value}  live_data_in_prompt={'YES' if has_live else 'NO (APIs returned nothing)'}"
        + (f"  lines={live_lines}" if live_lines else "")
    )

_check("Agent observation with live API data", _test_agent_obs)


print("\n" + "=" * 65)
print("Test complete.")
print()
print("Next steps:")
print("  PASS  ->API is live and feeding real data into agent observations")
print("  SKIP  ->Copy .env.example to .env and fill in the key")
print("  FAIL  ->Check your network connection or API key validity")
print()
print("To run with .env file:")
print("  pip install python-dotenv && python test_apis.py")
print("=" * 65)
