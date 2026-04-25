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
