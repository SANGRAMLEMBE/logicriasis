"""
World state management for LogiCrisis.
Holds the ground-truth supply network graph, cargo queue, and resource pool.
Agents NEVER see this directly — they receive filtered AgentObservation objects.
"""
from __future__ import annotations
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    AgentRole, Cargo, CargoType, Disruption, DisruptionType, Route,
)


# ── Network topology ──────────────────────────────────────────────────────────

NODES = ["Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore",
         "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat"]

EDGES: list[tuple[str, str, float]] = [
    ("Mumbai",    "Pune",      150),
    ("Mumbai",    "Ahmedabad", 530),
    ("Mumbai",    "Surat",     290),
    ("Delhi",     "Jaipur",    280),
    ("Delhi",     "Ahmedabad", 950),
    ("Kolkata",   "Hyderabad", 1500),
    ("Chennai",   "Bangalore", 350),
    ("Chennai",   "Hyderabad", 630),
    ("Bangalore", "Hyderabad", 570),
    ("Bangalore", "Pune",      840),
    ("Pune",      "Hyderabad", 560),
    ("Ahmedabad", "Jaipur",    670),
    ("Surat",     "Ahmedabad", 260),
]

REGIONS = {
    "West":  ["Mumbai", "Pune", "Surat", "Ahmedabad"],
    "North": ["Delhi", "Jaipur"],
    "East":  ["Kolkata"],
    "South": ["Chennai", "Bangalore", "Hyderabad"],
}


def _build_routes(rng: random.Random = None) -> dict[str, Route]:
    routes: dict[str, Route] = {}
    for (a, b, dist) in EDGES:
        # Capacity is distance-based (shorter roads = higher throughput). Deterministic.
        capacity = round(max(50.0, 220.0 - dist / 8.0), 1)
        rid = f"{a}-{b}"
        routes[rid] = Route(route_id=rid, from_node=a, to_node=b,
                            distance_km=dist, capacity_tons=capacity)
        rid2 = f"{b}-{a}"
        routes[rid2] = Route(route_id=rid2, from_node=b, to_node=a,
                             distance_km=dist, capacity_tons=capacity)
    return routes


def _region_of(node: str) -> str:
    for region, nodes in REGIONS.items():
        if node in nodes:
            return region
    return "Unknown"


# ── Agent resource state (hidden per-agent private state) ─────────────────────

@dataclass
class AgentState:
    agent_id: str
    role: AgentRole
    region: str
    capacity_tons: float
    budget: float
    trucks: int
    cold_storage_units: int = 0
    reputation: float = 1.0     # 0-2, affects bid success probability
    coalition_id: Optional[str] = None
    active_contracts: list[dict] = field(default_factory=list)


# ── Coalition record ──────────────────────────────────────────────────────────

@dataclass
class Coalition:
    coalition_id: str
    members: list[str]          # agent_ids
    lead: str                   # agent_id
    cargo_ids: list[str] = field(default_factory=list)
    reward_split: dict[str, float] = field(default_factory=dict)
    formed_turn: int = 0
    dissolved: bool = False


# ── Bid / contract record ─────────────────────────────────────────────────────

@dataclass
class Bid:
    bid_id: str
    from_agent: str
    to_agent: str
    cargo_id: str
    price: float
    capacity: float
    turn_issued: int
    accepted: bool = False
    rejected: bool = False
    breached: bool = False


# ── World state ───────────────────────────────────────────────────────────────

class WorldState:
    def __init__(
        self,
        curriculum_level: int = 1,
        seed: Optional[int] = None,
        cargo_count: Optional[int] = None,
        disruption_count: Optional[int] = None,
        max_turns: Optional[int] = None,
        cold_chain_ratio: float = 0.0,   # 0=random, 1.0=all cold chain
        capacity_multiplier: float = 1.0,  # scale agent starting capacity (0.3=scarce)
        deadline_max: Optional[int] = None, # cap all cargo deadlines (JIT scenarios)
        priority_weights: bool = False,     # assign priority 1-4 by cargo_type
    ):
        self.rng = random.Random(seed)
        self.curriculum_level = curriculum_level
        self.turn: int = 0
        self.max_turns: int = max_turns if max_turns is not None else 20

        self._cargo_count = cargo_count
        self._disruption_count = disruption_count
        self._cold_chain_ratio = cold_chain_ratio
        self._capacity_multiplier = capacity_multiplier
        self._deadline_max = deadline_max
        self._priority_weights = priority_weights

        self.routes: dict[str, Route] = _build_routes()
        self.cargo_queue: dict[str, Cargo] = {}
        self.disruptions: list[Disruption] = []
        self.agent_states: dict[str, AgentState] = {}
        self.coalitions: dict[str, Coalition] = {}
        self.bids: dict[str, Bid] = {}
        self.delivered_cargo: list[str] = []
        self.failed_cargo: list[str] = []
        self.audit_log: list[dict] = []
        self._route_heal_at: dict[str, int] = {}   # route_id -> turn when guaranteed heal fires

    # ── Initialisation ────────────────────────────────────────────────────────

    def reset(self, agent_ids: list[str], roles: list[AgentRole]) -> None:
        self.turn = 0
        self.routes = _build_routes()
        self.cargo_queue = {}
        self.disruptions = []
        self.agent_states = {}
        self.coalitions = {}
        self.bids = {}
        self.delivered_cargo = []
        self.failed_cargo = []
        self.audit_log = []

        self._route_heal_at = {}
        self._assign_agents(agent_ids, roles)
        self._generate_cargo()
        self._inject_disruptions()

    def _assign_agents(self, agent_ids: list[str], roles: list[AgentRole]) -> None:
        region_list = list(REGIONS.keys())
        for i, (aid, role) in enumerate(zip(agent_ids, roles)):
            region = region_list[i % len(region_list)]
            self.agent_states[aid] = AgentState(
                agent_id=aid,
                role=role,
                region=region,
                capacity_tons=self.rng.uniform(80, 200) * self._capacity_multiplier,
                budget=self.rng.uniform(5000, 15000),
                trucks=self.rng.randint(3, 10),
                cold_storage_units=self.rng.randint(0, 3),
            )

    def _generate_cargo(self) -> None:
        n_cargo = self._cargo_count if self._cargo_count is not None \
            else {1: 5, 2: 15, 3: 20}.get(self.curriculum_level, 5)
        nodes = NODES
        for i in range(n_cargo):
            # Step 1: cargo type (same call as original)
            if self._cold_chain_ratio >= 1.0:
                ctype = CargoType.COLD_CHAIN
                _ = self.rng.choice(list(CargoType))   # consume call to keep RNG sequence identical
            elif self._cold_chain_ratio > 0 and self.rng.random() < self._cold_chain_ratio:
                ctype = CargoType.COLD_CHAIN
            else:
                ctype = self.rng.choice(list(CargoType))

            # Steps 2-7: same rng call order as original
            value       = self.rng.uniform(500, 5000)
            raw_deadline = self.rng.randint(5, self.max_turns)
            origin      = self.rng.choice(nodes)
            destination = self.rng.choice(nodes)
            weight      = self.rng.uniform(1, 30)
            owner       = self.rng.choice(list(self.agent_states.keys()))

            deadline = min(raw_deadline, self._deadline_max) \
                if self._deadline_max is not None else raw_deadline

            priority = {
                CargoType.URGENT: 4, CargoType.COLD_CHAIN: 3,
                CargoType.STANDARD: 2, CargoType.BULK: 1,
            }.get(ctype, 1) if self._priority_weights else 1

            cargo = Cargo(
                cargo_id=f"C{i:03d}",
                cargo_type=ctype,
                value=value,
                deadline=deadline,
                origin=origin,
                destination=destination,
                weight_tons=weight,
                temp_sensitive=(ctype == CargoType.COLD_CHAIN),
                owner_agent=owner,
                priority=priority,
            )
            self.cargo_queue[cargo.cargo_id] = cargo

    def _inject_disruptions(self) -> None:
        n_dis = self._disruption_count if self._disruption_count is not None \
            else self.curriculum_level
        for _ in range(n_dis):
            dtype = self.rng.choice(list(DisruptionType))
            severity = self.rng.randint(1, min(self.curriculum_level * 2, 5))
            affected_nodes = self.rng.sample(NODES, k=min(severity, len(NODES)))
            affected_routes = [
                rid for rid, r in self.routes.items()
                if r.from_node in affected_nodes or r.to_node in affected_nodes
            ]
            # Block some routes proportional to severity
            routes_to_block = self.rng.sample(
                affected_routes, k=min(severity * 2, len(affected_routes))
            )
            for rid in routes_to_block:
                self.routes[rid].blocked = True
                self._route_heal_at[rid] = self.turn + self.rng.randint(3, 8)

            self.disruptions.append(Disruption(
                disruption_type=dtype,
                severity=severity,
                affected_nodes=affected_nodes,
                affected_routes=affected_routes,
                turns_remaining=self.max_turns,
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_open_routes(self) -> list[Route]:
        return [r for r in self.routes.values() if not r.blocked]

    def get_disrupted_route_ids(self) -> list[str]:
        return [r.route_id for r in self.routes.values() if r.blocked]

    def get_disrupted_nodes(self) -> list[str]:
        nodes: set[str] = set()
        for d in self.disruptions:
            nodes.update(d.affected_nodes)
        return list(nodes)

    def get_cargo_for_agent(self, agent_id: str) -> list[Cargo]:
        return [
            c for c in self.cargo_queue.values()
            if c.owner_agent == agent_id and not c.delivered and not c.spoiled
        ]

    def get_pending_deadlines(self, agent_id: str) -> list[tuple[str, int]]:
        result = []
        for c in self.get_cargo_for_agent(agent_id):
            turns_left = c.deadline - self.turn
            result.append((c.cargo_id, turns_left))
        return result

    def advance_turn(self) -> None:
        self.turn += 1
        # Tick disruptions
        for d in self.disruptions:
            d.turns_remaining -= 1
        # Stochastic route recovery
        recovered = []
        for rid, heal_at in list(self._route_heal_at.items()):
            if not self.routes[rid].blocked:
                recovered.append(rid)
                continue
            if self.turn >= heal_at or self.rng.random() < 0.15:
                self.routes[rid].blocked = False
                recovered.append(rid)
                self.log({"type": "route_recovered", "route_id": rid,
                          "early": self.turn < heal_at})
        for rid in recovered:
            del self._route_heal_at[rid]
        # Expire overdue cold-chain cargo (spoilage)
        for cargo in self.cargo_queue.values():
            if cargo.temp_sensitive and not cargo.delivered:
                if self.turn > cargo.deadline:
                    cargo.spoiled = True

    def log(self, event: dict) -> None:
        event["turn"] = self.turn
        self.audit_log.append(event)

    def otif_percent(self) -> float:
        total = len(self.cargo_queue)
        if total == 0:
            return 100.0
        on_time = sum(
            1 for c in self.cargo_queue.values()
            if c.delivered and not c.spoiled
        )
        return round(on_time / total * 100, 1)

    def severity_multiplier(self) -> float:
        if not self.disruptions:
            return 1.0
        avg_sev = sum(d.severity for d in self.disruptions) / len(self.disruptions)
        return 1.0 + avg_sev * 0.2

    def snapshot(self) -> dict:
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "disruptions": [
                {
                    "type": d.disruption_type.value,
                    "severity": d.severity,
                    "affected_nodes": d.affected_nodes,
                    "turns_remaining": d.turns_remaining,
                }
                for d in self.disruptions
            ],
            "cargo_summary": {
                "total": len(self.cargo_queue),
                "delivered": len(self.delivered_cargo),
                "failed": len(self.failed_cargo),
                "spoiled": sum(1 for c in self.cargo_queue.values() if c.spoiled),
            },
            "blocked_routes": self.get_disrupted_route_ids(),
            "otif_percent": self.otif_percent(),
            "coalitions": {
                cid: {"members": coal.members, "lead": coal.lead}
                for cid, coal in self.coalitions.items()
                if not coal.dissolved
            },
            "audit_log_len": len(self.audit_log),
        }
