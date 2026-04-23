"""
Benchmark script — measures OTIF improvement across N episodes.
Run BEFORE training (baseline) and AFTER training to produce the before/after demo.

Usage:
    python benchmark.py --episodes 50 --level 1
    python benchmark.py --episodes 50 --level 1 --adapter ./outputs/final
"""
from __future__ import annotations
import argparse
import json
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import LogiCrisisEnv, AgentAction, ActionType


def run_heuristic_episode(curriculum_level: int, seed: int) -> dict:
    """Baseline heuristic agent — always tries to reroute or waits."""
    env = LogiCrisisEnv(curriculum_level=curriculum_level, seed=seed)
    observations = env.reset()

    for _turn in range(env.world.max_turns):
        actions: dict[str, AgentAction] = {}
        for agent_id, obs in observations.items():
            action = _heuristic_action(env, agent_id, obs)
            actions[agent_id] = action

        result = env.step(actions)
        observations = result.observations
        if result.terminated or result.truncated:
            break

    snap = env.state()
    return {
        "seed": seed,
        "turns_taken": snap["turn"],
        "otif_percent": snap["otif_percent"],
        "delivered": snap["cargo_summary"]["delivered"],
        "total_cargo": snap["cargo_summary"]["total"],
        "spoiled": snap["cargo_summary"]["spoiled"],
        "coalitions_formed": len(snap["coalitions"]),
        "total_reward": sum(result.rewards.values()),
    }


def _heuristic_action(env: LogiCrisisEnv, agent_id: str, obs) -> AgentAction:
    """Simple greedy heuristic for baseline comparison."""
    if not obs.own_cargo_queue:
        return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT)

    cargo_id = obs.own_cargo_queue[0]
    cargo = env.world.cargo_queue.get(cargo_id)
    if not cargo:
        return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT)

    # Try to find an open route toward the destination
    for rid, route in env.world.routes.items():
        if not route.blocked and route.to_node == cargo.destination:
            return AgentAction(
                agent_id=agent_id,
                action_type=ActionType.REROUTE,
                cargo_id=cargo_id,
                route_id=rid,
                reasoning="greedy reroute toward destination",
            )

    # If cold-chain at risk, deploy cold storage
    if cargo.temp_sensitive and obs.own_capacity_tons > 0:
        state = env.world.agent_states.get(agent_id)
        if state and state.cold_storage_units > 0:
            return AgentAction(
                agent_id=agent_id,
                action_type=ActionType.DEPLOY_COLD_STORAGE,
                cargo_id=cargo_id,
                reasoning="cold chain at risk",
            )

    return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                       reasoning="no valid route found")


def run_benchmark(n_episodes: int, curriculum_level: int) -> dict:
    results = []
    for seed in range(n_episodes):
        ep = run_heuristic_episode(curriculum_level=curriculum_level, seed=seed)
        results.append(ep)
        if (seed + 1) % 10 == 0:
            avg_otif = sum(r["otif_percent"] for r in results) / len(results)
            print(f"  Episode {seed+1}/{n_episodes} | Avg OTIF so far: {avg_otif:.1f}%")

    avg_otif = sum(r["otif_percent"] for r in results) / len(results)
    avg_turns = sum(r["turns_taken"] for r in results) / len(results)
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    coalition_rate = sum(1 for r in results if r["coalitions_formed"] > 0) / len(results)
    cold_loss_rate = sum(r["spoiled"] for r in results) / max(sum(r["total_cargo"] for r in results), 1)

    summary = {
        "n_episodes": n_episodes,
        "curriculum_level": curriculum_level,
        "avg_otif_percent": round(avg_otif, 1),
        "avg_turns_to_complete": round(avg_turns, 1),
        "avg_total_reward": round(avg_reward, 3),
        "coalition_formation_rate": round(coalition_rate * 100, 1),
        "cold_chain_loss_rate_percent": round(cold_loss_rate * 100, 1),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiCrisis Benchmark")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    print(f"Running {args.episodes} heuristic baseline episodes at Level {args.level}…")
    summary = run_benchmark(args.episodes, args.level)

    print("\n=== BENCHMARK RESULTS ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")
