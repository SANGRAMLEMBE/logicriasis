"""
Task 3 — Cascade Failure Recovery (Hard)
Three cascading disruptions, 5 agents, 20 cargo, 20 turns.
Grader: 0.4 × OTIF + 0.3 × cold_chain + 0.2 × turn_efficiency + 0.1 × budget_efficiency
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task3CascadeFailureRecovery:
    id: str = "cascade_failure_recovery"
    name: str = "Cascade Failure Recovery"
    difficulty: str = "hard"
    description: str = (
        "Three cascading disruptions shut down 60% of routes. 5 agents must cooperate, "
        "negotiate SLAs, form coalitions, and protect cold-chain cargo across 20 deliveries."
    )
    max_turns: int = 20
    reward_range: list = None
    agents: int = 5
    cargo_count: int = 20
    disruptions: int = 3

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        env = LogiCrisisEnv(curriculum_level=3, agent_roster=DEFAULT_AGENTS[:5], seed=seed)
        return env

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return {"task_id": self.id, "score": 1.0, "otif_percent": 100.0,
                    "breakdown": {}, "passed": True, "verdict": "PASS"}

        # 1. OTIF (0.0–1.0)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # 2. Cold-chain (0.0–1.0)
        cold_items = [c for c in world.cargo_queue.values() if c.temp_sensitive]
        cold_score = (
            sum(1 for c in cold_items if c.delivered and not c.spoiled) / len(cold_items)
            if cold_items else 1.0
        )

        # 3. Turn efficiency: bonus for finishing faster (0.0–1.0)
        turns_used = world.turn
        turn_score = max(0.0, 1.0 - (turns_used / world.max_turns))

        # 4. Budget efficiency: how much budget remains (0.0–1.0)
        initial_budget = sum(100 * 100 for _ in world.agent_states)  # rough baseline
        remaining = sum(s.budget for s in world.agent_states.values())
        budget_score = min(remaining / max(initial_budget, 1), 1.0)

        score = round(
            0.4 * otif_score +
            0.3 * cold_score +
            0.2 * turn_score +
            0.1 * budget_score,
            4
        )
        passed = score >= 0.45   # hard task — lower pass threshold

        # Penalty if cascade failure (>60% cargo spoiled)
        spoiled_pct = sum(1 for c in world.cargo_queue.values() if c.spoiled) / total
        if spoiled_pct > 0.6:
            score = round(score * 0.5, 4)   # harsh cascade penalty
            passed = False

        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "otif_score": round(otif_score, 3),
                "cold_chain_score": round(cold_score, 3),
                "turn_efficiency_score": round(turn_score, 3),
                "budget_efficiency_score": round(budget_score, 3),
                "on_time_deliveries": on_time,
                "late_deliveries": late,
                "cold_items_intact": sum(1 for c in cold_items
                                         if c.delivered and not c.spoiled),
                "cold_items_total": len(cold_items),
                "cascade_penalty_applied": spoiled_pct > 0.6,
                "turns_used": turns_used,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }
