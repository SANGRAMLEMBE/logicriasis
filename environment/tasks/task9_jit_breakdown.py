"""
Task 9 — Just-In-Time Breakdown (Medium-Hard)
JIT manufacturing crisis: a major auto plant's supply chain collapses.
All cargo has ultra-tight deadlines (turns 3–6). Late = full penalty, no partial credit.
Agents must triage: decide which cargo to save and which to write off.

Optimal strategy: triage reasoning. A naive agent tries to deliver everything
and fails all deadlines. An intelligent agent picks the highest-value subset,
concentrates effort, and sacrifices the rest deliberately.

Grader: value-weighted strict OTIF (zero credit for late) + triage_efficiency
Pass threshold: 0.50
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task9JITBreakdown:
    id: str = "jit_breakdown"
    name: str = "Just-In-Time Breakdown"
    difficulty: str = "medium-hard"
    description: str = (
        "JIT manufacturing crisis: all 14 cargo items must arrive within turns 3–6 "
        "or the factory line halts (zero credit for late delivery). "
        "3 agents, 2 disruptions, 10 turns. Agents must triage — "
        "picking the highest-value subset to save and deliberately abandoning the rest."
    )
    max_turns: int = 10
    reward_range: list = None
    agents: int = 3
    cargo_count: int = 14
    disruptions: int = 2

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=2,
            agent_roster=DEFAULT_AGENTS[:3],
            seed=seed,
            cargo_count=14,
            disruption_count=2,
            max_turns=10,
            deadline_max=6,   # all deadlines capped at turn 6 (ultra-tight JIT)
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._empty_result(1.0)

        total_value = sum(c.value for c in world.cargo_queue.values())

        # Strict OTIF: ZERO credit for late delivery (JIT constraint)
        on_time_value = sum(
            c.value for c in world.cargo_queue.values()
            if c.delivered and not c.spoiled
            and c.delivered_turn != -1 and c.delivered_turn <= c.deadline
        )
        on_time_count = sum(
            1 for c in world.cargo_queue.values()
            if c.delivered and not c.spoiled
            and c.delivered_turn != -1 and c.delivered_turn <= c.deadline
        )

        value_score = on_time_value / max(total_value, 1)

        # Triage efficiency: did the agent concentrate on high-value cargo?
        # Score = delivered_value / max_achievable_value
        # Max achievable = sum of top-N cargo values where N ≈ what's deliverable in 6 turns
        sorted_values = sorted(
            [c.value for c in world.cargo_queue.values()], reverse=True
        )
        # Assume best case: 40% of cargo deliverable in 6 tight turns with 3 agents
        achievable_n = max(1, int(total * 0.4))
        max_achievable = sum(sorted_values[:achievable_n])
        triage_score = min(on_time_value / max(max_achievable, 1), 1.0)

        score = round(0.6 * value_score + 0.4 * triage_score, 4)
        passed = score >= 0.50

        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round((on_time_count / total) * 100, 1),
            "breakdown": {
                "value_score": round(value_score, 3),
                "triage_score": round(triage_score, 3),
                "on_time_deliveries": on_time_count,
                "late_deliveries": sum(1 for c in world.cargo_queue.values()
                                       if c.delivered and c.delivered_turn != -1
                                       and c.delivered_turn > c.deadline),
                "failed_deliveries": sum(1 for c in world.cargo_queue.values()
                                         if not c.delivered),
                "on_time_value": round(on_time_value, 0),
                "total_cargo_value": round(total_value, 0),
                "late_credit": 0,
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _empty_result(self, score):
        return {
            "task_id": self.id, "score": score, "otif_percent": 100.0,
            "breakdown": {}, "passed": score >= 0.50,
            "verdict": "PASS" if score >= 0.50 else "FAIL",
        }
