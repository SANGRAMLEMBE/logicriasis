"""
World state management for LogiCrisis.
Holds the ground-truth supply network graph, cargo queue, and resource pool.
Agents NEVER see this directly — they receive filtered AgentObservation objects.
"""
from __future__ import annotations
import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    AgentRole, Cargo, CargoType, Disruption, DisruptionType, Route,
)


# ── Load realistic data files ─────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_json(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_ROUTES_DATA      = _load_json("routes.json")
_CARGO_DATA       = _load_json("cargo_profiles.json")
_DISRUPTION_DATA  = _load_json("disruption_history.json")
_AGENT_DATA       = _load_json("agent_profiles.json")


# ── Network topology ──────────────────────────────────────────────────────────

NODES = ["Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore",
         "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat"]

REGIONS = {
    "West":  ["Mumbai", "Pune", "Surat", "Ahmedabad"],
    "North": ["Delhi", "Jaipur"],
    "East":  ["Kolkata"],
    "South": ["Chennai", "Bangalore", "Hyderabad"],
}

# Build EDGES from data file (real NH distances + traffic-based capacities)
EDGES: list[tuple[str, str, float, float]] = [
    (r["from"], r["to"], r["distance_km"], r["capacity_tons"])
    for r in _ROUTES_DATA["routes"]
]


def _build_routes(rng: random.Random = None) -> dict[str, Route]:
    routes: dict[str, Route] = {}
    for (a, b, dist, capacity) in EDGES:
        rid = f"{a}-{b}"
        routes[rid] = Route(route_id=rid, from_node=a, to_node=b,
                            distance_km=dist, capacity_tons=capacity)
        # Add reverse direction with same distance and capacity
        rid2 = f"{b}-{a}"
        if rid2 not in routes:
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
        for i, (aid, role) in enumerate(zip(agent_ids, roles)):
            profile = _AGENT_DATA.get(role.value, _AGENT_DATA["carrier"])
            home_regions = profile.get("home_regions", list(REGIONS.keys()))
            region = home_regions[i % len(home_regions)]

            cap_min, cap_max = profile["capacity_tons"]
            bud_min, bud_max = profile["budget"]
            trk_min, trk_max = profile["trucks"]
            csu_min, csu_max = profile["cold_storage_units"]

            self.agent_states[aid] = AgentState(
                agent_id=aid,
                role=role,
                region=region,
                capacity_tons=round(self.rng.uniform(cap_min, cap_max) * self._capacity_multiplier, 1),
                budget=round(self.rng.uniform(bud_min, bud_max), 0),
                trucks=self.rng.randint(trk_min, trk_max),
                cold_storage_units=self.rng.randint(csu_min, csu_max),
            )

    def _generate_cargo(self) -> None:
        n_cargo = self._cargo_count if self._cargo_count is not None \
            else {1: 5, 2: 15, 3: 20}.get(self.curriculum_level, 5)

        type_weights  = {k: v for k, v in _CARGO_DATA["type_weights"].items() if not k.startswith("_")}
        value_ranges  = _CARGO_DATA["value_range_usd"]
        weight_ranges = _CARGO_DATA["weight_range_tons"]
        deadline_ranges = _CARGO_DATA["deadline_range_turns"]
        od_pairs      = _CARGO_DATA["od_pairs"]
        type_keys = list(type_weights.keys())
        type_probs = list(type_weights.values())

        for i in range(n_cargo):
            # Cargo type: use realistic freight distribution weights
            if self._cold_chain_ratio >= 1.0:
                ctype = CargoType.COLD_CHAIN
            elif self._cold_chain_ratio > 0 and self.rng.random() < self._cold_chain_ratio:
                ctype = CargoType.COLD_CHAIN
            else:
                chosen = self.rng.choices(type_keys, weights=type_probs, k=1)[0]
                ctype = CargoType(chosen)

            tkey = ctype.value
            val_min, val_max   = value_ranges[tkey]
            wgt_min, wgt_max   = weight_ranges[tkey]
            ddl_min, ddl_max   = deadline_ranges[tkey]

            value  = round(self.rng.uniform(val_min, val_max), 2)
            weight = round(self.rng.uniform(wgt_min, wgt_max), 2)

            # Origin/destination: use realistic trade OD pairs (80%) or random (20%)
            if self.rng.random() < 0.80 and od_pairs.get(tkey):
                pair = self.rng.choice(od_pairs[tkey])
                origin      = pair["origin"]
                destination = pair["destination"]
            else:
                origin      = self.rng.choice(NODES)
                destination = self.rng.choice(NODES)

            raw_deadline = self.rng.randint(ddl_min, min(ddl_max, self.max_turns))
            deadline = min(raw_deadline, self._deadline_max) \
                if self._deadline_max is not None else raw_deadline

            owner = self.rng.choice(list(self.agent_states.keys()))

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

        _dtype_weights = {k: v for k, v in _DISRUPTION_DATA["disruption_type_weights"].items() if not k.startswith("_")}
        dtype_keys   = list(_dtype_weights.keys())
        dtype_probs  = list(_dtype_weights.values())
        flood_cities = _DISRUPTION_DATA["flood_prone_cities"]
        heal_ranges  = _DISRUPTION_DATA["severity_heal_turns"]

        for _ in range(n_dis):
            # Pick disruption type using historical frequency weights
            dtype_str = self.rng.choices(dtype_keys, weights=dtype_probs, k=1)[0]
            dtype = DisruptionType(dtype_str)

            # Severity capped by curriculum level
            max_sev = min(self.curriculum_level * 2, 5)

            if dtype == DisruptionType.FLOOD:
                # Pick cities weighted by their historical flood frequency
                freq_map = {"high": 3, "medium": 2, "low": 1, "rare": 0.5}
                flood_weights = [freq_map.get(flood_cities[c]["frequency"], 1) for c in NODES]
                severity_range = flood_cities.get(
                    self.rng.choices(NODES, weights=flood_weights, k=1)[0],
                    {}
                ).get("typical_severity", [1, max_sev])
                severity = min(self.rng.randint(*severity_range), max_sev)
                # High-frequency flood cities are more likely affected
                weighted_nodes = self.rng.choices(
                    NODES, weights=flood_weights, k=min(severity + 1, len(NODES))
                )
                affected_nodes = list(dict.fromkeys(weighted_nodes))[:severity]

            elif dtype == DisruptionType.PORT_STRIKE:
                # Port strikes hit port cities: Mumbai, Chennai, Kolkata
                port_cities = ["Mumbai", "Chennai", "Kolkata"]
                severity = self.rng.randint(2, max_sev)
                n_ports = min(severity // 2 + 1, len(port_cities))
                affected_nodes = self.rng.sample(port_cities, k=max(1, n_ports))

            else:  # ROAD_CLOSURE
                # Road closures: pick from historically closed routes
                closure_routes = _DISRUPTION_DATA["road_closure_history"]
                if closure_routes:
                    chosen = self.rng.choice(closure_routes)
                    route_cities = [chosen["route"].split("-")[0],
                                    chosen["route"].split("-")[1]]
                    affected_nodes = [c for c in route_cities if c in NODES]
                else:
                    affected_nodes = self.rng.sample(NODES, k=2)
                severity = self.rng.randint(1, max(1, max_sev - 1))

            affected_routes = [
                rid for rid, r in self.routes.items()
                if r.from_node in affected_nodes or r.to_node in affected_nodes
            ]
            routes_to_block = self.rng.sample(
                affected_routes, k=min(severity * 2, len(affected_routes))
            )

            # Heal time based on historical event durations
            heal_min, heal_max = heal_ranges.get(str(severity), [3, 8])
            for rid in routes_to_block:
                self.routes[rid].blocked = True
                self._route_heal_at[rid] = self.turn + self.rng.randint(heal_min, heal_max)

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

    def get_recovering_routes(self) -> list[str]:
        """Route IDs that are currently blocked but have a scheduled heal turn."""
        return [rid for rid in self._route_heal_at if self.routes[rid].blocked]

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
