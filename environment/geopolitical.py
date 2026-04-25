"""
Geopolitical event models for LogiCrisis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeopoliticalEvent:
    event_id: str
    event_type: str              # e.g. "trade_dispute", "sanction", "corridor_closure"
    affected_regions: list[str] = field(default_factory=list)
    affected_routes: list[str] = field(default_factory=list)
    severity: int = 1            # 1-5
    turns_active: int = 10
    description: str = ""
    issuing_agent: Optional[str] = None
    resolved: bool = False


class GeopoliticalState:
    """Tracks active geopolitical events and exposes per-region alert queries."""

    def __init__(self) -> None:
        self.events: list[GeopoliticalEvent] = []

    def tick(self) -> None:
        for ev in self.events:
            if not ev.resolved:
                ev.turns_active -= 1
                if ev.turns_active <= 0:
                    ev.resolved = True

    def alerts_for_region(self, region: str) -> list[str]:
        return [
            ev.description or ev.event_type
            for ev in self.events
            if not ev.resolved and region in ev.affected_regions
        ]
