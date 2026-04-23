"""
Task 2 — Coalition Logistics (Medium)
Two disruptions, 3 agents, 15 cargo (includes cold-chain), 15 turns.
Grader: 0.5 × OTIF + 0.3 × cold_chain_integrity + 0.2 × coalition_formed
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task2CoalitionLogistics:
    id: str = "coalition_logistics"
    name: str = "Coalition Logistics"
    difficulty: str = "medium"
    description: str = (
        "Two simultaneous disruptions with cold-chain cargo at risk. "
        "3 agents must form coalitions and deliver 15 cargo items within 15 turns."
    )
    max_turns: int = 15
    reward_range: list = None
    agents: int = 3
    cargo_count: int = 15
    disruptions: int = 2

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        roster = DEFAULT_AGENTS[:3]
        env = LogiCrisisEnv(curriculum_level=2, agent_roster=roster, seed=seed)
        return env

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._result(1.0, 100.0, 1.0, True)

        # OTIF component (0.0–1.0)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # Cold-chain component (0.0–1.0)
        cold_total = sum(1 for c in world.cargo_queue.values() if c.temp_sensitive)
        cold_intact = sum(1 for c in world.cargo_queue.values()
                         if c.temp_sensitive and c.delivered and not c.spoiled)
        cold_score = (cold_intact / cold_total) if cold_total > 0 else 1.0

        # Coalition component (binary 0 or 1)
        any_coalition = any(not c.dissolved for c in world.coalitions.values())
        coalition_score = 1.0 if any_coalition else 0.0

        # Weighted composite
        score = round(
            0.5 * otif_score + 0.3 * cold_score + 0.2 * coalition_score, 4
        )
        passed = score >= 0.55

        return {
            "task_id": self.id,
            "score": score,
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "otif_score": round(otif_score, 3),
                "cold_chain_score": round(cold_score, 3),
                "coalition_formed": any_coalition,
                "coalition_score": coalition_score,
                "on_time_deliveries": on_time,
                "cold_items_intact": cold_intact,
                "cold_items_total": cold_total,
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _result(self, score, otif_pct, cold_score, coalition):
        return {
            "task_id": self.id,
            "score": score,
            "otif_percent": otif_pct,
            "breakdown": {"cold_chain_score": cold_score, "coalition_formed": coalition},
            "passed": score >= 0.55,
            "verdict": "PASS" if score >= 0.55 else "FAIL",
        }
