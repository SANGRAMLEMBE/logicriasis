"""
Task 8 — Capacity Crunch (Hard)
COVID-surge inspired: sudden demand doubles but fleet capacity is at 25% due to
driver shortages, vehicle breakdowns, and fuel rationing. Agents must buy/sell
capacity from each other via the bid market to handle the surge.

Optimal strategy: market-based rebalancing. Agents with excess capacity sell it,
agents with cargo but no capacity buy it. A pure routing agent will deliver <40%
of cargo; a market-aware agent should reach 70%+.

Grader: 0.40 × OTIF + 0.35 × capacity_utilisation + 0.25 × market_activity
Pass threshold: 0.45
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task8CapacityCrunch:
    id: str = "capacity_crunch"
    name: str = "Capacity Crunch"
    difficulty: str = "hard"
    description: str = (
        "COVID-surge scenario: demand is high but fleet capacity is at 25% — "
        "drivers are unavailable, trucks are broken down. 5 agents must trade "
        "capacity via the bid market to rebalance the fleet and deliver 20 cargo. "
        "Pure routing fails; market coordination is the only winning strategy."
    )
    max_turns: int = 15
    reward_range: list = None
    agents: int = 5
    cargo_count: int = 20
    disruptions: int = 2

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=3,
            agent_roster=DEFAULT_AGENTS[:5],
            seed=seed,
            cargo_count=20,
            disruption_count=2,
            max_turns=15,
            capacity_multiplier=0.25,   # agents start at 25% normal capacity
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._empty_result(1.0)

        # OTIF (40%)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # Capacity utilisation (35%): weight delivered / total agent capacity used
        total_delivered_weight = sum(
            c.weight_tons for c in world.cargo_queue.values() if c.delivered
        )
        total_capacity = sum(s.capacity_tons for s in world.agent_states.values())
        # Starting capacity was 25% — use delivered weight vs cargo weight as proxy
        total_cargo_weight = sum(c.weight_tons for c in world.cargo_queue.values())
        utilisation_score = min(total_delivered_weight / max(total_cargo_weight, 1), 1.0)

        # Market activity (25%): accepted bids = capacity trades happened
        accepted_bids = sum(1 for b in world.bids.values() if b.accepted)
        total_bids    = len(world.bids)
        market_score  = min(accepted_bids / max(total_bids + 1, 4), 1.0)

        score = round(
            0.40 * otif_score +
            0.35 * utilisation_score +
            0.25 * market_score,
            4
        )
        passed = score >= 0.45

        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "otif_score": round(otif_score, 3),
                "utilisation_score": round(utilisation_score, 3),
                "market_score": round(market_score, 3),
                "on_time_deliveries": on_time,
                "late_deliveries": late,
                "accepted_bids": accepted_bids,
                "total_bids": total_bids,
                "delivered_weight_tons": round(total_delivered_weight, 1),
                "total_cargo_weight_tons": round(total_cargo_weight, 1),
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _empty_result(self, score):
        return {
            "task_id": self.id, "score": score, "otif_percent": 100.0,
            "breakdown": {}, "passed": score >= 0.45,
            "verdict": "PASS" if score >= 0.45 else "FAIL",
        }
