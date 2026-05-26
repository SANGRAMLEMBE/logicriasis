"""
Per-agent memory system for LogiCrisis autonomous agents.

Short-term: last N action outcomes within the current episode.
Long-term:  lessons extracted at episode end, persisted across episodes.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TurnRecord:
    turn: int
    action_type: str
    reasoning: str
    outcome: str   # "success" | "failed" | "pending"
    reward: float


class AgentMemory:
    SHORT_TERM_LIMIT = 8
    LONG_TERM_LIMIT  = 12

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._short_term: list[TurnRecord] = []
        self._long_term: list[str] = []
        self.episode_count = 0

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def episode_start(self):
        self._short_term.clear()

    def episode_end(self, score: float):
        self.episode_count += 1

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_turn(self, turn: int, action_type: str, reasoning: str, reward: float):
        outcome = "success" if reward > 0 else ("failed" if reward < 0 else "neutral")
        self._short_term.append(TurnRecord(turn, action_type, reasoning, outcome, reward))
        if len(self._short_term) > self.SHORT_TERM_LIMIT:
            self._short_term.pop(0)

    def add_lesson(self, lesson: str):
        self._long_term.append(lesson.strip()[:120])
        if len(self._long_term) > self.LONG_TERM_LIMIT:
            self._long_term.pop(0)

    # ── Read ──────────────────────────────────────────────────────────────────

    def short_term_context(self) -> str:
        if not self._short_term:
            return "(no prior turns this episode)"
        lines = []
        for r in self._short_term[-6:]:
            lines.append(
                f"  Turn {r.turn}: {r.action_type} → {r.outcome} "
                f"(reward={r.reward:+.2f}) | {r.reasoning[:70]}"
            )
        return "\n".join(lines)

    def long_term_context(self) -> str:
        if not self._long_term:
            return "(no lessons from past episodes yet)"
        return "\n".join(f"  • {lesson}" for lesson in self._long_term[-6:])

    def to_prompt_block(self) -> str:
        return (
            "=== MY ACTION HISTORY (this episode) ===\n"
            + self.short_term_context()
            + "\n\n=== LESSONS FROM PAST EPISODES ===\n"
            + self.long_term_context()
        )
