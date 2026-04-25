"""
LogiCrisis inference script — OpenEnv hackathon baseline.

Runs all 3 tasks using an LLM (via OpenAI-compatible API) as the agent policy.
Falls back to a deterministic heuristic if the LLM is unavailable or parsing fails.

Environment variables:
  API_BASE_URL  — base URL for OpenAI-compatible API (default: https://api.openai.com/v1)
  MODEL_NAME    — model to query (default: gpt-4o-mini)
  HF_TOKEN      — optional HuggingFace token (passed as Bearer if API_BASE_URL points to HF)
  OPENAI_API_KEY — API key (also accepted as fallback for HF_TOKEN)

Output format (stdout):
  [START] {...json...}
  [STEP]  {...json...}   (one per agent-turn)
  [END]   {...json...}
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

# Allow running from root without install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from environment import LogiCrisisEnv, AgentAction, ActionType
from environment.tasks import ALL_TASK_IDS, get_task
from environment.world import WorldState


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL     = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN         = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "")

MAX_RETRIES  = 2    # LLM call retries before falling back to heuristic
SEED         = 42


# ── OpenAI client ─────────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "sk-no-key",
        )
    return _client


# ── Structured log helpers ────────────────────────────────────────────────────

def _log(tag: str, payload: dict) -> None:
    """Emit one structured log line to stdout."""
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


# ── Heuristic fallback policy ─────────────────────────────────────────────────

def _heuristic_action(agent_id: str, obs, world: WorldState) -> dict:
    """
    Deterministic greedy policy:
    - For each cargo owned, find a non-blocked direct route to its destination.
    - If found, reroute; else wait.
    - For medium/hard tasks, propose a coalition on turn 1 if no coalition exists.
    """
    state = world.agent_states.get(agent_id)
    if state is None:
        return {"action_type": "wait", "reasoning": "no agent state"}

    # Coalition formation on first turn (helps Task 2 & 3 graders)
    if world.turn == 0 and not state.coalition_id:
        other_agents = [aid for aid in world.agent_states if aid != agent_id]
        if len(other_agents) >= 1:
            members = other_agents[:2]
            split = {agent_id: 0.5}
            for m in members:
                split[m] = 0.5 / len(members)
            return {
                "action_type": "propose_coalition",
                "coalition_id": f"coal_{agent_id}",
                "coalition_members": members,
                "reward_split": split,
                "reasoning": "early coalition for collaborative delivery",
            }

    # Cold storage rescue — highest priority
    for cargo in world.cargo_queue.values():
        if (cargo.temp_sensitive and not cargo.spoiled and not cargo.delivered
                and cargo.owner_agent == agent_id and state.cold_storage_units > 0
                and state.budget >= 200):
            return {
                "action_type": "deploy_cold_storage",
                "cargo_id": cargo.cargo_id,
                "reasoning": "deploy cold storage to protect temp-sensitive cargo",
            }

    # Reroute cargo to destination
    for cargo in world.cargo_queue.values():
        if cargo.delivered or cargo.spoiled or cargo.owner_agent != agent_id:
            continue
        for route in world.routes.values():
            if route.blocked:
                continue
            if route.to_node == cargo.destination:
                if state.capacity_tons >= cargo.weight_tons:
                    return {
                        "action_type": "reroute",
                        "cargo_id": cargo.cargo_id,
                        "route_id": route.route_id,
                        "reasoning": f"direct route {route.route_id} → {cargo.destination}",
                    }

    return {"action_type": "wait", "reasoning": "no actionable cargo or routes"}


# ── LLM policy ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a logistics agent in a multi-agent supply chain crisis simulation.
You must respond with a single valid JSON action object.

Available action_types:
  reroute, request_transfer, prioritize_cargo, deploy_cold_storage,
  make_bid, accept_bid, reject_bid, counter_propose,
  propose_coalition, join_coalition, leave_coalition, assign_coalition_role, wait

Required fields: action_type (string), reasoning (string, max 100 chars)
Optional fields depend on action_type:
  reroute              → cargo_id, route_id
  make_bid             → cargo_id, bid_price, bid_capacity, target_agent
  accept_bid/reject_bid → bid_id
  counter_propose      → bid_id, bid_price
  propose_coalition    → coalition_id, coalition_members (list), reward_split (dict)
  join_coalition       → coalition_id
  deploy_cold_storage  → cargo_id
  prioritize_cargo     → cargo_id

Always output exactly one JSON object, nothing else.
Example: {"action_type": "reroute", "cargo_id": "C000", "route_id": "Mumbai-Pune", "reasoning": "direct route to destination"}
"""


def _llm_action(agent_id: str, prompt_text: str) -> Optional[dict]:
    """Call LLM and parse JSON action. Returns None on failure."""
    client = _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text},
                ],
                max_tokens=256,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON from possible markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw)
            if "action_type" not in action:
                continue
            return action
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
    return None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool = True) -> dict:
    """
    Run one full episode for a given task.
    Returns the grader result dict.
    """
    task = get_task(task_id)
    env = task.make_env(seed=SEED)
    observations = env.reset()
    agent_ids = list(observations.keys())

    _log("START", {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "agent_ids": agent_ids,
        "max_turns": task.max_turns,
        "cargo_count": task.cargo_count,
        "disruptions": task.disruptions,
        "model": MODEL_NAME if use_llm else "heuristic",
        "seed": SEED,
    })

    episode_rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}
    turn = 0

    while True:
        actions: dict[str, AgentAction] = {}
        chosen_actions: dict[str, dict] = {}

        for agent_id, obs in observations.items():
            # Try LLM first, fall back to heuristic
            action_dict = None
            if use_llm:
                action_dict = _llm_action(agent_id, obs.to_prompt_text())

            if action_dict is None:
                action_dict = _heuristic_action(agent_id, obs, env.world)

            chosen_actions[agent_id] = action_dict

            # Convert dict → AgentAction
            try:
                atype = ActionType(action_dict.get("action_type", "wait"))
            except ValueError:
                atype = ActionType.WAIT

            actions[agent_id] = AgentAction(
                agent_id=agent_id,
                action_type=atype,
                cargo_id=action_dict.get("cargo_id"),
                route_id=action_dict.get("route_id"),
                target_region=action_dict.get("target_region"),
                bid_price=action_dict.get("bid_price"),
                bid_capacity=action_dict.get("bid_capacity"),
                target_agent=action_dict.get("target_agent"),
                bid_id=action_dict.get("bid_id"),
                coalition_id=action_dict.get("coalition_id"),
                coalition_members=action_dict.get("coalition_members"),
                coalition_role=action_dict.get("coalition_role"),
                reward_split=action_dict.get("reward_split"),
                reasoning=str(action_dict.get("reasoning", ""))[:120],
            )

        result = env.step(actions)
        turn += 1

        for agent_id, reward in result.rewards.items():
            episode_rewards[agent_id] = episode_rewards.get(agent_id, 0.0) + reward

        _log("STEP", {
            "turn": turn,
            "actions": {
                aid: {
                    "action_type": chosen_actions[aid].get("action_type"),
                    "reasoning":   chosen_actions[aid].get("reasoning", "")[:80],
                }
                for aid in agent_ids
            },
            "rewards": {aid: round(r, 4) for aid, r in result.rewards.items()},
            "otif_percent": round(result.info.get("otif_percent", 0.0), 1),
            "terminated": result.terminated,
            "truncated": result.truncated,
        })

        observations = result.observations

        if result.terminated or result.truncated:
            break

    grade = task.grade(env)

    _log("END", {
        "task_id": task_id,
        "score": grade["score"],
        "otif_percent": grade["otif_percent"],
        "passed": grade["passed"],
        "verdict": grade["verdict"],
        "breakdown": grade.get("breakdown", {}),
        "cumulative_rewards": {aid: round(r, 4) for aid, r in episode_rewards.items()},
        "turns_used": turn,
    })

    return grade


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    use_llm = bool(HF_TOKEN)

    if not use_llm:
        print(
            "[INFO] No API key found (HF_TOKEN / OPENAI_API_KEY). "
            "Running heuristic baseline.",
            flush=True,
        )

    results = []
    total_start = time.monotonic()

    for task_id in ALL_TASK_IDS:
        grade = run_episode(task_id, use_llm=use_llm)
        results.append(grade)

    elapsed = time.monotonic() - total_start

    # Final summary
    scores = [r["score"] for r in results]
    print("\n" + "=" * 60, flush=True)
    print("LogiCrisis Baseline Summary", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['task_id']:<30} score={r['score']:.4f}  "
            f"OTIF={r['otif_percent']:.1f}%  {status}",
            flush=True,
        )
    print(f"\n  Average score: {sum(scores)/len(scores):.4f}", flush=True)
    print(f"  Elapsed:       {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
