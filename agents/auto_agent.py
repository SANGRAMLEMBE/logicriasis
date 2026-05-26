"""
AutoAgent: base class for all LogiCrisis autonomous specialist agents.

Each agent runs a two-step loop per turn:
  1. think()  — free-form reasoning about the current situation (no JSON)
  2. act()    — convert that reasoning into a structured AgentAction JSON

At episode end, reflect() extracts a short lesson into long-term memory.
"""
from __future__ import annotations
import json
import re
from typing import Optional

from environment import AgentAction, ActionType
from agents.memory import AgentMemory
from agents.prompts import get_system_prompt, get_allowed_actions
from agents.role_configs import get_role_config


_THINK_SYSTEM = (
    "You are the {title} in LogiCrisis. "
    "Think step by step about the current crisis — do NOT output JSON yet.\n"
    "Reason through:\n"
    "  1. What is the most urgent problem right now?\n"
    "  2. What are my options given my role and allowed actions?\n"
    "  3. What will each option lead to in 1-2 turns?\n"
    "  4. What should I do and why?\n"
    "Keep your analysis to 3-5 sentences."
)

_ACT_INTRO = (
    "You are the {title} in LogiCrisis. "
    "Based on your analysis, output exactly ONE JSON action. "
    "Output ONLY the JSON object — nothing else.\n\n"
)

_REFLECT_SYSTEM = (
    "You are the {title} reflecting on a completed LogiCrisis episode. "
    "Write ONE lesson (max 20 words) that would help your future self perform better. "
    "Focus on the single most impactful timing or action decision."
)


class AutoAgent:
    """
    Base autonomous agent. Subclasses set `role` as a class attribute.
    Call decide(obs, turn) each step; call on_episode_end() when done.
    """

    role: str = "carrier"

    def __init__(self, agent_id: str, engine):
        self.agent_id = agent_id
        self.engine = engine
        self.memory = AgentMemory(agent_id)
        self._cfg = get_role_config(self.role)
        self._system_prompt = get_system_prompt(self.role)
        self._allowed = get_allowed_actions(self.role)
        self.last_thought: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_episode_start(self):
        self.memory.episode_start()
        self.last_thought = ""

    def on_episode_end(self, score: float):
        self.memory.episode_end(score)
        lesson = self._reflect(score)
        if lesson:
            self.memory.add_lesson(lesson)

    # ── Main decision interface ───────────────────────────────────────────────

    def decide(self, obs, turn: int = 0) -> AgentAction:
        """Think → Act. Returns a valid AgentAction for this turn."""
        self.last_thought = self._think(obs, turn)
        action_dict = self._act(self.last_thought, obs)
        return self._build_action(action_dict)

    def record_result(self, turn: int, action: AgentAction, reward: float):
        """Call after env.step() to store outcome in short-term memory."""
        self.memory.record_turn(
            turn=turn,
            action_type=action.action_type.value,
            reasoning=action.reasoning or "",
            reward=reward,
        )

    # ── Thinking step ─────────────────────────────────────────────────────────

    def _think(self, obs, turn: int) -> str:
        title = self._cfg.get("title", self.role)
        system = _THINK_SYSTEM.format(title=title)

        user = (
            f"Turn {turn}. Current situation:\n\n"
            + obs.to_prompt_text()
            + "\n\n"
            + self.memory.to_prompt_block()
            + "\n\nAnalyze the situation. What should you do this turn?"
        )

        return self.engine.generate(
            system_prompt=system,
            user_prompt=user,
            max_tokens=220,
            temperature=0.45,
        ).strip()

    # ── Action step ───────────────────────────────────────────────────────────

    def _act(self, thought: str, obs) -> dict:
        title = self._cfg.get("title", self.role)
        system = _ACT_INTRO.format(title=title) + self._system_prompt

        user = (
            "My analysis:\n"
            + thought
            + "\n\nSituation summary:\n"
            + obs.to_prompt_text()[:700]
            + "\n\nOutput your JSON action:"
        )

        raw = self.engine.generate(
            system_prompt=system,
            user_prompt=user,
            max_tokens=260,
            temperature=0.15,
        )
        return self._parse_json(raw)

    # ── Reflection step ───────────────────────────────────────────────────────

    def _reflect(self, score: float) -> Optional[str]:
        title = self._cfg.get("title", self.role)
        system = _REFLECT_SYSTEM.format(title=title)

        user = (
            f"Episode score: {score:.3f}\n"
            f"My action history:\n{self.memory.short_term_context()}\n\n"
            "What is the single most important lesson to remember next time?"
        )

        try:
            lesson = self.engine.generate(
                system_prompt=system,
                user_prompt=user,
                max_tokens=45,
                temperature=0.3,
            ).strip()
            return lesson[:120] if lesson else None
        except Exception:
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict:
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"action_type": "wait", "reasoning": "json parse error — waiting"}

    def _build_action(self, d: dict) -> AgentAction:
        action_str = d.get("action_type", "wait")

        # Reject actions outside this role's allowed set
        if action_str not in self._allowed:
            action_str = "wait"
            d["reasoning"] = f"action not allowed for {self.role}, defaulting to wait"

        try:
            atype = ActionType(action_str)
        except ValueError:
            atype = ActionType.WAIT

        return AgentAction(
            agent_id=self.agent_id,
            action_type=atype,
            cargo_id=d.get("cargo_id"),
            route_id=d.get("route_id"),
            target_region=d.get("target_region"),
            bid_price=d.get("bid_price"),
            bid_capacity=d.get("bid_capacity"),
            target_agent=d.get("target_agent"),
            bid_id=d.get("bid_id"),
            coalition_id=d.get("coalition_id"),
            coalition_members=d.get("coalition_members"),
            coalition_role=d.get("coalition_role"),
            reward_split=d.get("reward_split"),
            reasoning=str(d.get("reasoning", ""))[:120],
        )
