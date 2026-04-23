"""
Task 4 — Cold Chain Emergency (Medium-Hard)
Pharmaceutical crisis: 12 cargo, ALL temp-sensitive, 3 agents, 12 turns, 2 disruptions.
Grader: 0.7 × cold_chain_integrity + 0.3 × OTIF
Pass threshold: 0.60
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task4ColdChainEmergency:
    id: str = "cold_chain_emergency"
    name: str = "Cold Chain Emergency"
    difficulty: str = "medium-hard"
    description: str = (
        "A pharmaceutical crisis: ALL 12 cargo items are temperature-sensitive. "
        "Agents must deploy cold storage, prioritise delivery order, and coordinate "
        "before spoilage occurs. 2 disruptions, 12 turns — no room for waste."
    )
    max_turns: int = 12
    reward_range: list = None
    agents: int = 3
    cargo_count: int = 12
    disruptions: int = 2

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=2,
            agent_roster=DEFAULT_AGENTS[:3],
            seed=seed,
            cargo_count=12,
            disruption_count=2,
            max_turns=12,
            cold_chain_ratio=1.0,   # ALL cargo is temp-sensitive
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._result(1.0, 1.0, 1.0)

        # Cold-chain integrity (primary — 70% weight)
        cold_items = [c for c in world.cargo_queue.values() if c.temp_sensitive]
        cold_delivered = sum(1 for c in cold_items if c.delivered and not c.spoiled)
        cold_score = cold_delivered / len(cold_items) if cold_items else 1.0

        # OTIF (secondary — 30% weight)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # Heavy spoilage penalty: if >50% spoiled, score halved
        spoiled_pct = sum(1 for c in cold_items if c.spoiled) / max(len(cold_items), 1)

        score = round(0.7 * cold_score + 0.3 * otif_score, 4)
        if spoiled_pct > 0.5:
            score = round(score * 0.5, 4)

        passed = score >= 0.60
        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "cold_chain_score": round(cold_score, 3),
                "otif_score": round(otif_score, 3),
                "cold_items_delivered": cold_delivered,
                "cold_items_total": len(cold_items),
                "cold_items_spoiled": sum(1 for c in cold_items if c.spoiled),
                "spoilage_penalty_applied": spoiled_pct > 0.5,
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _result(self, score, cold_score, otif_score):
        return {
            "task_id": self.id,
            "score": score,
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {"cold_chain_score": cold_score, "otif_score": otif_score},
            "passed": score >= 0.60,
            "verdict": "PASS" if score >= 0.60 else "FAIL",
        }
