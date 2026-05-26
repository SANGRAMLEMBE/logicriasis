"""
LogiCrisis — Autonomous Agent Runner

Loads the GRPO-trained LoRA adapter and runs all 6 specialist agents
autonomously through the 9 crisis tasks. Each agent thinks before acting
and learns lessons across episodes.

Usage:
  python run_agents.py                               # all 9 tasks, curriculum order
  python run_agents.py --task task1_single_route_recovery
  python run_agents.py --mode adaptive --episodes 18
  python run_agents.py --adapter WIZARDIAN/logicriasis-adapter --quiet

Environment variables:
  ADAPTER_REPO   — HuggingFace adapter repo (overrides --adapter)
  HF_TOKEN       — HuggingFace token (needed to pull private adapter)
"""
from __future__ import annotations
import argparse
import os
import sys

# Force UTF-8 stdout/stderr on Windows so unicode chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env before anything else so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; keys can also be set in the shell

HF_HOME_DEFAULT = os.environ.get("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_HOME", HF_HOME_DEFAULT)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME_DEFAULT, "hub"))
os.makedirs(HF_HOME_DEFAULT, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="LogiCrisis Autonomous Agent Runner")
    parser.add_argument(
        "--task", default="all",
        help="Task ID to run, or 'all' for full benchmark (default: all)",
    )
    parser.add_argument(
        "--mode", default="curriculum",
        choices=["curriculum", "adaptive", "single"],
        help="Run mode: curriculum | adaptive | single (default: curriculum)",
    )
    parser.add_argument(
        "--episodes", type=int, default=18,
        help="Number of episodes for adaptive mode (default: 18)",
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("ADAPTER_REPO", "WIZARDIAN/logicriasis-adapter"),
        help="HuggingFace adapter repo ID",
    )
    parser.add_argument(
        "--base", default="unsloth/llama-3-8b-instruct-bnb-4bit",
        help="Base model repo ID",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-turn logs (only show final scores)",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print("[INIT] Loading LoRA adapter...")
    from agents.model_engine import get_engine
    engine = get_engine(adapter_repo=args.adapter, base_model=args.base)

    # ── Build orchestrator ────────────────────────────────────────────────────
    from agents.orchestrator import MultiAgentOrchestrator
    orchestrator = MultiAgentOrchestrator(engine=engine, verbose=not args.quiet)

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.mode == "adaptive":
        orchestrator.run_adaptive(n_episodes=args.episodes, seed=args.seed)

    elif args.mode == "single" or (args.task != "all"):
        task_id = args.task if args.task != "all" else "task1_single_route_recovery"
        orchestrator.run_episode(task_id=task_id, seed=args.seed)

    else:  # curriculum (default)
        orchestrator.run_all_tasks(seed=args.seed)


if __name__ == "__main__":
    main()
