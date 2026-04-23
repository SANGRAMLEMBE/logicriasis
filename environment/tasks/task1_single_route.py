"""
Task 1 — Single Route Recovery (Easy)
One disruption, 2 agents, 5 cargo, 10 turns.
Grader: score = on_time_deliveries / total_cargo (0.0–1.0)
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv
from ..models import AgentRole


@dataclass
class Task1SingleRouteRecovery:
    id: str = "single_route_recovery"
    name: str = "Single Route Recovery"
    difficulty: str = "easy"
    description: str = (
        "One disruption blocks key routes. 2 agents must reroute 5 cargo items "
        "to their destinations within 10 turns."
    )
    max_turns: int = 10
    reward_range: list = None
    agents: int = 2
    cargo_count: int = 5
    disruptions: int = 1

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        from ..env import DEFAULT_AGENTS
        roster = DEFAULT_AGENTS[:2]
        env = LogiCrisisEnv(curriculum_level=1, agent_roster=roster, seed=seed)
        return env

    def grade(self, env: LogiCrisisEnv) -> dict:
        """
        Deterministic grader: score = on_time_deliveries / total_cargo
        Partial credit given for late deliveries (0.5 each).
        """
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._result(1.0, 0, 0, 0, "No cargo to deliver")

        on_time = sum(
            1 for c in world.cargo_queue.values()
            if c.delivered and not c.spoiled
            and c.delivered_turn != -1 and c.delivered_turn <= c.deadline
        )
        late = sum(
            1 for c in world.cargo_queue.values()
            if c.delivered and not c.spoiled
            and c.delivered_turn != -1 and c.delivered_turn > c.deadline
        )
        failed = sum(
            1 for c in world.cargo_queue.values()
            if c.spoiled or (not c.delivered and world.turn >= world.max_turns)
        )

        # Partial credit: on_time=1.0pt, late=0.5pt, failed=0pt
        raw = (on_time * 1.0 + late * 0.5) / total
        score = round(min(raw, 1.0), 4)
        passed = score >= 0.6

        return self._result(
            score, on_time, late, failed,
            verdict="PASS" if passed else "FAIL",
            extra={"turns_used": world.turn, "otif": world.otif_percent()}
        )

    def _result(self, score, on_time, late, failed, verdict="", extra=None):
        return {
            "task_id": self.id,
            "score": score,
            "otif_percent": round(score * 100, 1),
            "breakdown": {
                "on_time_deliveries": on_time,
                "late_deliveries": late,
                "failed_deliveries": failed,
                **(extra or {}),
            },
            "passed": score >= 0.6,
            "verdict": verdict or ("PASS" if score >= 0.6 else "FAIL"),
        }
