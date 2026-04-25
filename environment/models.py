"""
Data models for LogiCrisis environment.
Action / Observation / State dataclasses following OpenEnv spec.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    CARRIER = "carrier"
    WAREHOUSE = "warehouse"
    CUSTOMS_BROKER = "customs_broker"
    INSURER = "insurer"
    SHIPPER = "shipper"


class DisruptionType(str, Enum):
    FLOOD = "flood"
    PORT_STRIKE = "port_strike"
    ROAD_CLOSURE = "road_closure"


class ActionType(str, Enum):
    # Logistics
    REROUTE = "reroute"
    REQUEST_TRANSFER = "request_transfer"
    PRIORITIZE_CARGO = "prioritize_cargo"
    DEPLOY_COLD_STORAGE = "deploy_cold_storage"
    # Negotiation
    MAKE_BID = "make_bid"
    ACCEPT_BID = "accept_bid"
    REJECT_BID = "reject_bid"
    COUNTER_PROPOSE = "counter_propose"
    # Coalition
    PROPOSE_COALITION = "propose_coalition"
    JOIN_COALITION = "join_coalition"
    LEAVE_COALITION = "leave_coalition"
    ASSIGN_COALITION_ROLE = "assign_coalition_role"
    # No-op
    WAIT = "wait"
    # Geopolitical
    ISSUE_GEOPOLITICAL_ALERT = "issue_geopolitical_alert"
    NEGOTIATE_TRADE_CORRIDOR = "negotiate_trade_corridor"
    APPLY_SANCTIONS = "apply_sanctions"
    REQUEST_DIPLOMATIC_BYPASS = "request_diplomatic_bypass"


class CargoType(str, Enum):
    STANDARD = "standard"
    COLD_CHAIN = "cold_chain"   # pharma / food
    URGENT = "urgent"
    BULK = "bulk"


# ── Core dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Cargo:
    cargo_id: str
    cargo_type: CargoType
    value: float            # monetary value (hidden from non-owning agents)
    deadline: int           # turn by which it must be delivered
    origin: str
    destination: str
    weight_tons: float
    temp_sensitive: bool = False
    delivered: bool = False
    spoiled: bool = False
    owner_agent: str = ""
    priority: int = 1           # 1=low, 2=medium, 3=high, 4=critical
    delivered_turn: int = -1    # turn on which delivery was completed (-1 = not delivered)


@dataclass
class Route:
    route_id: str
    from_node: str
    to_node: str
    distance_km: float
    capacity_tons: float
    blocked: bool = False
    congestion_level: float = 0.0   # 0-1


@dataclass
class Disruption:
    disruption_type: DisruptionType
    severity: int           # 1-5
    affected_nodes: list[str] = field(default_factory=list)
    affected_routes: list[str] = field(default_factory=list)
    turns_remaining: int = 20


@dataclass
class AgentAction:
    agent_id: str
    action_type: ActionType
    # Logistics fields
    cargo_id: Optional[str] = None
    route_id: Optional[str] = None
    target_region: Optional[str] = None
    # Negotiation fields
    bid_price: Optional[float] = None
    bid_capacity: Optional[float] = None
    target_agent: Optional[str] = None
    bid_id: Optional[str] = None
    # Coalition fields
    coalition_id: Optional[str] = None
    coalition_members: Optional[list[str]] = None
    coalition_role: Optional[str] = None
    reward_split: Optional[dict[str, float]] = None
    # Reasoning (shown in demo, used for process supervision)
    reasoning: str = ""


@dataclass
class AgentObservation:
    """Partial observation returned to each agent."""
    agent_id: str
    role: AgentRole
    turn: int
    max_turns: int

    # Own state
    own_region: str
    own_capacity_tons: float
    own_budget: float
    own_cargo_queue: list[str]          # cargo_ids visible to this agent
    pending_deadlines: list[tuple[str, int]]  # (cargo_id, turns_left)

    # Public disruption info (exact severity hidden)
    disrupted_routes: list[str]
    disrupted_nodes: list[str]

    # Noisy neighbor signals
    neighbor_bids: list[dict]           # recent public bids from neighbors
    coalition_proposals: list[dict]     # active coalition invitations

    # Own history (last 3 actions + outcomes)
    action_history: list[dict]

    # Active contracts / coalitions
    active_coalition_id: Optional[str] = None
    active_contracts: list[dict] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = [
            f"=== Agent {self.agent_id} | Role: {self.role.value} | Turn {self.turn}/{self.max_turns} ===",
            f"Region: {self.own_region}",
            f"Capacity: {self.own_capacity_tons:.1f} tons | Budget: ${self.own_budget:.0f}",
            f"Cargo queue: {self.own_cargo_queue}",
            f"Pending deadlines: {self.pending_deadlines}",
            f"Disrupted routes: {self.disrupted_routes}",
            f"Disrupted nodes: {self.disrupted_nodes}",
            f"Neighbor bids: {self.neighbor_bids}",
            f"Coalition proposals: {self.coalition_proposals}",
            f"Action history: {self.action_history}",
            f"Active coalition: {self.active_coalition_id}",
            f"Active contracts: {self.active_contracts}",
        ]
        return "\n".join(lines)


@dataclass
class StepResult:
    observations: dict[str, AgentObservation]   # agent_id -> observation
    rewards: dict[str, float]                    # agent_id -> reward
    reward_breakdown: dict[str, dict]            # agent_id -> {R1..R6}
    terminated: bool
    truncated: bool
    info: dict
