"""
Task 5 — Negotiation Sprint (Medium)
4 agents must bid, counter-propose, and form coalitions to move 10 cargo in 10 turns.
No grader credit for solo routing — all reward comes from cooperative mechanisms.
Grader: 0.35 × OTIF + 0.40 × negotiation_activity + 0.25 × coalition_quality
Pass threshold: 0.50
"""
from __future__ import annotations
from dataclasses import dataclass
from ..env import LogiCrisisEnv, DEFAULT_AGENTS


@dataclass
class Task5NegotiationSprint:
    id: str = "negotiation_sprint"
    name: str = "Negotiation Sprint"
    difficulty: str = "medium"
    description: str = (
        "4 agents, 10 cargo, 10 turns, 1 disruption. Every cargo starts with the "
        "wrong agent — they MUST negotiate transfers and form coalitions. "
        "Solo routing yields minimal score; cooperation is the only winning strategy."
    )
    max_turns: int = 10
    reward_range: list = None
    agents: int = 4
    cargo_count: int = 10
    disruptions: int = 1

    def __post_init__(self):
        self.reward_range = [0.0, 1.0]

    def make_env(self, seed: int = 42) -> LogiCrisisEnv:
        return LogiCrisisEnv(
            curriculum_level=2,
            agent_roster=DEFAULT_AGENTS[:4],
            seed=seed,
            cargo_count=10,
            disruption_count=1,
            max_turns=10,
        )

    def grade(self, env: LogiCrisisEnv) -> dict:
        world = env.world
        total = len(world.cargo_queue)
        if total == 0:
            return self._empty_result(1.0)

        # OTIF (35%)
        on_time = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn <= c.deadline)
        late    = sum(1 for c in world.cargo_queue.values()
                      if c.delivered and not c.spoiled
                      and c.delivered_turn != -1 and c.delivered_turn > c.deadline)
        otif_score = min((on_time * 1.0 + late * 0.5) / total, 1.0)

        # Negotiation activity (40%): count accepted bids + counter-proposals
        accepted_bids  = sum(1 for b in world.bids.values() if b.accepted)
        total_bids     = len(world.bids)
        counters       = sum(1 for ev in world.audit_log
                             if ev.get("type") == "counter_proposed")
        # Score based on activity: ≥5 accepted bids = full score
        negotiation_score = min((accepted_bids * 1.0 + counters * 0.5) / max(total_bids + 1, 5), 1.0)

        # Coalition quality (25%): active + at least 2 members + delivered cargo
        coalitions_active = [c for c in world.coalitions.values() if not c.dissolved]
        coal_score = 0.0
        if coalitions_active:
            best = max(len(c.members) for c in coalitions_active)
            coal_score = min(best / 3.0, 1.0)   # full score for 3+ member coalition

        score = round(0.35 * otif_score + 0.40 * negotiation_score + 0.25 * coal_score, 4)
        passed = score >= 0.50

        return {
            "task_id": self.id,
            "score": min(score, 1.0),
            "otif_percent": round(otif_score * 100, 1),
            "breakdown": {
                "otif_score": round(otif_score, 3),
                "negotiation_score": round(negotiation_score, 3),
                "coalition_score": round(coal_score, 3),
                "accepted_bids": accepted_bids,
                "total_bids": total_bids,
                "counter_proposals": counters,
                "active_coalitions": len(coalitions_active),
                "on_time_deliveries": on_time,
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
