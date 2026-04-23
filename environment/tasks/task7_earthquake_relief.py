"""
Task 7 — Earthquake Relief Operations (Hard)
Real-world inspired: South India earthquake aftermath (modeled on 2004 Indian Ocean
tsunami response logistics). 18 cargo with 4 priority tiers. Agents must deliver
CRITICAL items first even if it means sub-optimal routing.

Optimal strategy: LLM must reason about priority ordering under constraint,
not just delivery speed. A naive routing agent will score ~0.3; a priority-aware
agent should score 0.7+.

Grader: priority-weighted OTIF
  CRITICAL (medical) = 4× weight, penalty -5 if undelivered
  HIGH (rescue equipment) = 2× weight
  MEDIUM (food/water) = 1× weight
  LOW (shelter materials) = 0.5× weight
Pass threshold: 0.55
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS
from ..models import CargoType


@dataclass
class Task7EarthquakeRelief:
    id: str = "earthquake_relief"
    name: str = "Earthquake Relief Operations"
    difficulty: str = "hard"
    description: str = (
        "South India earthquake aftermath: 18 cargo items across 4 priority tiers "
        "(CRITICAL medical → LOW shelter). 4 agents, 3 disruptions, 15 turns. "
        "Naive routing fails — agents must reason about humanitarian priority ordering. "
        "Undelivered CRITICAL cargo incurs severe penalty."
    )
    max_turns: int = 15
    reward_range: list = None
    agents: int = 4
    cargo_count: int = 18
    disruptions: int = 3

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=3,
            agent_roster=DEFAULT_AGENTS[:4],
            seed=seed,
            cargo_count=18,
            disruption_count=3,
            max_turns=15,
            priority_weights=True,   # URGENT=4, COLD_CHAIN=3, STANDARD=2, BULK=1
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._empty_result(1.0)

        WEIGHT = {4: 4.0, 3: 2.0, 2: 1.0, 1: 0.5}
        LABEL  = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}

        by_priority: dict[int, dict] = {p: {"total": 0, "delivered": 0, "on_time": 0}
                                         for p in [1, 2, 3, 4]}

        for c in world.cargo_queue.values():
            p = c.priority
            by_priority[p]["total"] += 1
            if c.delivered and not c.spoiled:
                by_priority[p]["delivered"] += 1
                if c.delivered_turn != -1 and c.delivered_turn <= c.deadline:
                    by_priority[p]["on_time"] += 1

        # Weighted score: on_time=1pt, late=0.5pt, each weighted by priority
        max_possible = sum(
            WEIGHT[p] * by_priority[p]["total"] for p in [1, 2, 3, 4]
        )
        earned = 0.0
        for p in [1, 2, 3, 4]:
            d = by_priority[p]
            on_time = d["on_time"]
            late = d["delivered"] - on_time
            earned += WEIGHT[p] * (on_time * 1.0 + late * 0.5)

        base_score = earned / max(max_possible, 1)

        # Critical penalty: -0.15 per undelivered CRITICAL item
        undelivered_critical = (
            by_priority[4]["total"] - by_priority[4]["delivered"]
        )
        penalty = undelivered_critical * 0.15
        score = round(max(0.0, min(base_score - penalty, 1.0)), 4)
        passed = score >= 0.55

        return {
            "task_id": self.id,
            "score": score,
            "otif_percent": round(base_score * 100, 1),
            "breakdown": {
                "weighted_score": round(base_score, 3),
                "critical_penalty": round(penalty, 3),
                **{
                    f"{LABEL[p]}_delivered": f"{by_priority[p]['delivered']}/{by_priority[p]['total']}"
                    for p in [4, 3, 2, 1]
                },
                "undelivered_critical": undelivered_critical,
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _empty_result(self, score):
        return {
            "task_id": self.id, "score": score, "otif_percent": 100.0,
            "breakdown": {}, "passed": score >= 0.55,
            "verdict": "PASS" if score >= 0.55 else "FAIL",
        }
