"""
MultiAgentOrchestrator — runs all 6 autonomous agents through LogiCrisis episodes.

Three run modes:
  curriculum  — all 9 tasks in order (standard benchmark)
  adaptive    — score-driven replay: agents replay tasks where they scored lowest
  single      — one specific task

Agents persist across episodes so long-term memory accumulates across runs.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

from environment import LogiCrisisEnv, ActionType, AgentAction
from environment.tasks import ALL_TASK_IDS, get_task
from agents.specialist_agents import make_agent
from agents.auto_agent import AutoAgent


@dataclass
class EpisodeResult:
    task_id: str
    score: float
    otif_percent: float
    passed: bool
    verdict: str
    turns_used: int
    agent_rewards: dict[str, float]
    elapsed_sec: float
    breakdown: dict = field(default_factory=dict)


def _infer_role(agent_id: str) -> str:
    """'geo_analyst_0' -> 'geopolitical_analyst', 'carrier_0' -> 'carrier', etc."""
    _MAP = {
        "geo_analyst":    "geopolitical_analyst",
        "customs_broker": "customs_broker",
        "warehouse":      "warehouse",
        "carrier":        "carrier",
        "insurer":        "insurer",
        "shipper":        "shipper",
    }
    base = "_".join(agent_id.split("_")[:-1]) if "_" in agent_id else agent_id
    return _MAP.get(base, base)


class MultiAgentOrchestrator:
    """
    Creates and manages specialist agents, runs autonomous episodes,
    and accumulates long-term memory across the full session.
    """

    def __init__(self, engine, verbose: bool = True):
        self.engine = engine
        self.verbose = verbose
        self._agents: dict[str, AutoAgent] = {}   # persisted across episodes
        self.results: list[EpisodeResult] = []

    # -- Agent management ------------------------------------------------------

    def _get_agents(self, task_id: str) -> dict[str, AutoAgent]:
        """
        Return agents for this task's roster, creating new ones only for
        agent_ids we haven't seen before. Existing agents keep their memory.
        """
        task = get_task(task_id)
        probe_env = task.make_env(seed=0)
        probe_env.reset()
        roster = list(probe_env.world.agent_states.keys())

        agents: dict[str, AutoAgent] = {}
        for agent_id in roster:
            if agent_id not in self._agents:
                role = _infer_role(agent_id)
                self._agents[agent_id] = make_agent(agent_id, role, self.engine)
                if self.verbose:
                    print(f"  [INIT] Created {agent_id} ({role})")
            agents[agent_id] = self._agents[agent_id]
        return agents

    # -- Episode runner --------------------------------------------------------

    def run_episode(self, task_id: str, seed: int = 42) -> EpisodeResult:
        task = get_task(task_id)
        env = task.make_env(seed=seed)

        agents = self._get_agents(task_id)
        for agent in agents.values():
            agent.on_episode_start()

        observations = env.reset()
        cumulative_rewards: dict[str, float] = {aid: 0.0 for aid in observations}
        turn = 0
        t0 = time.time()

        self._header(task_id, task, agents)

        while True:
            turn += 1
            actions: dict[str, AgentAction] = {}

            for agent_id, obs in observations.items():
                agent = agents.get(agent_id)
                if agent is None:
                    actions[agent_id] = AgentAction(
                        agent_id=agent_id,
                        action_type=ActionType.WAIT,
                        reasoning="unassigned agent",
                    )
                    continue

                if self.verbose:
                    print(f"  [T{turn:02d}] {agent_id:20s} thinking...", end=" ", flush=True)

                action = agent.decide(obs, turn=turn)
                actions[agent_id] = action

                if self.verbose:
                    print(
                        f"-> {action.action_type.value:<22} "
                        f"| {(action.reasoning or '')[:55]}"
                    )

            step_result = env.step(actions)

            for agent_id, agent in agents.items():
                reward = step_result.rewards.get(agent_id, 0.0)
                cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + reward
                if agent_id in actions:
                    agent.record_result(turn, actions[agent_id], reward)

            otif = step_result.info.get("otif_percent", 0.0)
            if self.verbose:
                rew_str = "  ".join(
                    f"{k}:{v:+.2f}" for k, v in step_result.rewards.items()
                )
                print(f"       OTIF={otif:.1f}%  rewards=[{rew_str}]")

            observations = step_result.observations

            if step_result.terminated or step_result.truncated:
                reason = "all cargo resolved" if step_result.terminated else "max turns"
                if self.verbose:
                    print(f"  [END] Episode finished: {reason}")
                break

        grade = task.grade(env)
        score = grade["score"]

        for agent in agents.values():
            agent.on_episode_end(score)

        result = EpisodeResult(
            task_id=task_id,
            score=score,
            otif_percent=grade.get("otif_percent", 0.0),
            passed=grade.get("passed", False),
            verdict=grade.get("verdict", "FAIL"),
            turns_used=turn,
            agent_rewards=cumulative_rewards,
            elapsed_sec=time.time() - t0,
            breakdown=grade.get("breakdown", {}),
        )
        self.results.append(result)
        self._summary_line(result)
        return result

    # -- Multi-episode runners -------------------------------------------------

    def run_all_tasks(self, seed: int = 42) -> list[EpisodeResult]:
        """Standard benchmark: run all 9 tasks in curriculum order."""
        self._banner("Autonomous Agent Tournament — All 9 Tasks")
        results = [self.run_episode(t, seed=seed) for t in ALL_TASK_IDS]
        self._final_summary(results)
        return results

    def run_adaptive(self, n_episodes: int = 18, seed: int = 42) -> list[EpisodeResult]:
        """
        Adaptive curriculum: agents start with all 9 tasks in order, then
        replay whichever task they scored lowest on. This lets agents
        'choose' where to improve — the environment adapts to agent weakness.
        """
        self._banner(f"Adaptive Curriculum — {n_episodes} Episodes")
        task_scores: dict[str, list[float]] = {t: [] for t in ALL_TASK_IDS}
        queue = list(ALL_TASK_IDS)
        results = []

        for i in range(n_episodes):
            if queue:
                task_id = queue.pop(0)
            else:
                # Replay the task with the lowest mean score
                task_id = min(
                    task_scores,
                    key=lambda t: (
                        sum(task_scores[t]) / len(task_scores[t])
                        if task_scores[t] else 0.0
                    ),
                )

            print(f"\n[Episode {i+1:02d}/{n_episodes}]  task={task_id}")
            result = self.run_episode(task_id, seed=seed + i)
            results.append(result)
            task_scores[task_id].append(result.score)

        self._final_summary(results)
        return results

    # -- Display ---------------------------------------------------------------

    def _banner(self, title: str):
        if self.verbose:
            print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    def _header(self, task_id: str, task, agents: dict):
        if not self.verbose:
            return
        print(f"\n{'-' * 60}")
        print(f"  Task     : {task_id}")
        print(f"  Agents   : {list(agents.keys())}")
        print(f"  Max turns: {task.max_turns}  |  Cargo: {task.cargo_count}  |  "
              f"Disruptions: {task.disruptions}")
        print(f"{'-' * 60}")

    def _summary_line(self, r: EpisodeResult):
        if not self.verbose:
            return
        status = "PASS OK" if r.passed else "FAIL X"
        print(
            f"\n  Score={r.score:.4f}  OTIF={r.otif_percent:.1f}%  "
            f"{status}  turns={r.turns_used}  time={r.elapsed_sec:.1f}s"
        )

    def _final_summary(self, results: list[EpisodeResult]):
        if not results:
            return
        print(f"\n{'=' * 60}")
        print("Final Results")
        print(f"{'=' * 60}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.task_id:<38} {r.score:.4f}  {status}")
        scores = [r.score for r in results]
        passed = sum(1 for r in results if r.passed)
        print(f"\n  Average score : {sum(scores)/len(scores):.4f}")
        print(f"  Tasks passed  : {passed}/{len(results)}")
        print("=" * 60)
