"""
Six specialist AutoAgent subclasses — one per LogiCrisis role.
Each class sets its role so the base class picks up the right prompts,
reward weights, and allowed action set automatically.
"""
from __future__ import annotations
from agents.auto_agent import AutoAgent


class CarrierAgent(AutoAgent):
    """
    Focuses on OTIF% and truck utilisation.
    Reroutes immediately when routes are blocked; sells idle capacity via bids.
    """
    role = "carrier"


class WarehouseAgent(AutoAgent):
    """
    Owns cold chain integrity (R4 × 3.0 weight).
    Pre-deploys cold storage on temperature alerts — never waits until spoilage.
    """
    role = "warehouse"


class CustomsBrokerAgent(AutoAgent):
    """
    Manages trade corridors and tariff negotiations.
    Acts on ExchangeRate / GDELT signals before shipments are stranded at borders.
    """
    role = "customs_broker"


class InsurerAgent(AutoAgent):
    """
    Drives bid market activity and coalition ROI.
    Counter-proposes instead of rejecting — every bid closed counts toward grader score.
    """
    role = "insurer"


class ShipperAgent(AutoAgent):
    """
    Triages CRITICAL → COLD_CHAIN → URGENT → STANDARD cargo by deadline pressure.
    Buys capacity from Carriers when overwhelmed rather than waiting.
    """
    role = "shipper"


class GeoAnalystAgent(AutoAgent):
    """
    Issues corridor alerts 1-2 turns ahead of disruptions to earn shared_bonus.
    Negotiates trade corridors and applies sanctions when GDELT severity is high.
    """
    role = "geopolitical_analyst"


# ── Registry ──────────────────────────────────────────────────────────────────

_ROLE_TO_CLASS: dict[str, type[AutoAgent]] = {
    "carrier":              CarrierAgent,
    "warehouse":            WarehouseAgent,
    "customs_broker":       CustomsBrokerAgent,
    "insurer":              InsurerAgent,
    "shipper":              ShipperAgent,
    "geopolitical_analyst": GeoAnalystAgent,
}


def make_agent(agent_id: str, role: str, engine) -> AutoAgent:
    """
    Instantiate the correct specialist class for a given role.
    Falls back to base AutoAgent if role is unrecognised.
    """
    cls = _ROLE_TO_CLASS.get(role, AutoAgent)
    agent = cls(agent_id=agent_id, engine=engine)
    return agent
