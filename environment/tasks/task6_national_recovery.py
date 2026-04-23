"""
Task 6 — Full National Recovery (Expert / Very Hard)
4 simultaneous disruptions, all 5 agents, 25 cargo (40% cold-chain), 25 turns.
The hardest task — tests everything: OTIF, cold chain, coalition, negotiation, budget.
Grader: 0.30 × OTIF + 0.25 × cold_chain + 0.20 × coalition + 0.15 × negotiation + 0.10 × budget
Cascade penalty: if >50% cargo spoiled, score ×= 0.4
Pass threshold: 0.35 (very hard)
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task6NationalRecovery:
    id: str = "national_recovery"
    name: str = "Full National Recovery"
    difficulty: str = "expert"
    description: str = (
        "India's entire logistics network is failing: 4 simultaneous disruptions, "
        "25 cargo (40% pharmaceutical cold-chain), 5 agents, 25 turns. "
        "Agents must master every mechanic — coalition, negotiation, cold storage, "
        "budget efficiency — to score above 0.35. The expert benchmark."
    )
    max_turns: int = 25
    reward_range: list = None
    agents: int = 5
    cargo_count: int = 25
    disruptions: int = 4

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=3,
            agent_roster=DEFAULT_AGENTS[:5],
            seed=seed,
            cargo_count=25,
            disruption_count=4,
            max_turns=25,
            cold_chain_ratio=0.4,   # 40% of cargo is cold-chain
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._empty_result(1.0)

        # 1. OTIF (30%)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # 2. Cold chain (25%)
        cold_items = [c for c in world.cargo_queue.values() if c.temp_sensitive]
        cold_score = (
            sum(1 for c in cold_items if c.delivered and not c.spoiled) / len(cold_items)
            if cold_items else 1.0
        )

        # 3. Coalition quality (20%)
        active_coalitions = [c for c in world.coalitions.values() if not c.dissolved]
        if active_coalitions:
            max_members = max(len(c.members) for c in active_coalitions)
            coal_score = min(max_members / 4.0, 1.0)   # full score = 4-member coalition
        else:
            coal_score = 0.0

        # 4. Negotiation (15%)
        accepted_bids = sum(1 for b in world.bids.values() if b.accepted)
        negotiation_score = min(accepted_bids / 5.0, 1.0)   # full score = 5+ accepted bids

        # 5. Budget efficiency (10%)
        initial_budget = sum(100 * 100 for _ in world.agent_states)
        remaining = sum(s.budget for s in world.agent_states.values())
        budget_score = min(remaining / max(initial_budget, 1), 1.0)

        score = round(
            0.30 * otif_score +
            0.25 * cold_score +
            0.20 * coal_score +
            0.15 * negotiation_score +
            0.10 * budget_score,
            4
        )

        # Cascade penalty: >50% spoiled = brutal reduction
        spoiled_pct = sum(1 for c in world.cargo_queue.values() if c.spoiled) / total
        cascade_applied = spoiled_pct > 0.5
        if cascade_applied:
            score = round(score * 0.4, 4)

        passed = score >= 0.35

        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "otif_score": round(otif_score, 3),
                "cold_chain_score": round(cold_score, 3),
                "coalition_score": round(coal_score, 3),
                "negotiation_score": round(negotiation_score, 3),
                "budget_score": round(budget_score, 3),
                "on_time_deliveries": on_time,
                "late_deliveries": late,
                "cold_items_intact": sum(1 for c in cold_items if c.delivered and not c.spoiled),
                "cold_items_total": len(cold_items),
                "active_coalitions": len(active_coalitions),
                "accepted_bids": accepted_bids,
                "cascade_penalty_applied": cascade_applied,
                "turns_used": world.turn,
            },
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
        }

    def _empty_result(self, score):
        return {
            "task_id": self.id, "score": score, "otif_percent": 100.0,
            "breakdown": {}, "passed": score >= 0.35,
            "verdict": "PASS" if score >= 0.35 else "FAIL",
        }
