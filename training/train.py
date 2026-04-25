"""
TRL GRPO training loop for LogiCrisis — deep specialist manager agents.

Each agent role trains with:
- A role-filtered action space (only domain-relevant actions)
- Role-weighted reward multipliers (Carrier: R1×2.0, Warehouse: R4×3.0, etc.)
- Role-specific curriculum datasets

Run:
    python -m training.train --level 1 --episodes 200

Hackathon guide refs: sections 10, 11, 16, 17.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/Colab

import torch
from datasets import Dataset


# ── Lazy imports (handle missing deps gracefully for env-only testing) ─────────

def _import_unsloth():
    try:
        from unsloth import FastLanguageModel
        return FastLanguageModel
    except ImportError:
        raise SystemExit(
            "[ERROR] unsloth not installed.\n"
            "Install with: pip install unsloth"
        )


def _import_trl():
    try:
        from trl import GRPOConfig, GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except ImportError:
        raise SystemExit(
            "[ERROR] trl not installed.\n"
            "Install with: pip install trl>=0.9.0"
        )


# ── Environment imports ────────────────────────────────────────────────────────

from environment import LogiCrisisEnv, AgentAction, ActionType
from agents.prompts import get_system_prompt, build_user_prompt, get_allowed_actions
from agents.role_configs import compute_role_weighted_reward, get_role_config


# ── Role-aware action parser ───────────────────────────────────────────────────

def _parse_llm_action(agent_id: str, text: str, role: str = "") -> AgentAction:
    """Parse LLM JSON output into AgentAction. Validates against role's allowed actions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    try:
        data = json.loads(text)
        action_str = data.get("action_type", "wait")

        # Validate action against role's allowed list
        allowed = get_allowed_actions(role) if role else None
        if allowed and action_str not in allowed:
            action_str = "wait"   # out-of-domain action → fall back to wait

        atype = ActionType(action_str)
        return AgentAction(
            agent_id=agent_id,
            action_type=atype,
            cargo_id=data.get("cargo_id"),
            route_id=data.get("route_id"),
            target_region=data.get("target_region"),
            bid_price=data.get("bid_price"),
            bid_capacity=data.get("bid_capacity"),
            target_agent=data.get("target_agent"),
            bid_id=data.get("bid_id"),
            coalition_id=data.get("coalition_id"),
            coalition_members=data.get("coalition_members"),
            coalition_role=data.get("coalition_role"),
            reward_split=data.get("reward_split"),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                           reasoning="parse_error")


# ── Episode rollout ────────────────────────────────────────────────────────────

def run_episode(
    model,
    tokenizer,
    curriculum_level: int = 1,
    seed: int = 42,
    max_new_tokens: int = 256,
) -> list[dict]:
    """
    Run one full episode. Returns list of (prompt, response, reward) records
    that TRL's GRPOTrainer can consume.
    """
    env = LogiCrisisEnv(curriculum_level=curriculum_level, seed=seed)
    observations = env.reset()
    trajectory: list[dict] = []

    for _turn in range(env.world.max_turns):
        actions: dict[str, AgentAction] = {}

        for agent_id, obs in observations.items():
            role = obs.role.value
            messages = [
                {"role": "system", "content": get_system_prompt(role)},
                {"role": "user",   "content": build_user_prompt(obs.to_prompt_text())},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            action = _parse_llm_action(agent_id, response_text, role=role)
            actions[agent_id] = action

            trajectory.append({
                "prompt": text,
                "response": response_text,
                "agent_id": agent_id,
                "role": role,
                "turn": _turn,
            })

        result = env.step(actions)
        observations = result.observations

        # Attach role-weighted rewards to trajectory entries
        n_agents = len(actions)
        for i, (agent_id, _) in enumerate(actions.items()):
            idx = len(trajectory) - n_agents + i
            breakdown = result.reward_breakdown.get(agent_id, {})
            role = trajectory[idx]["role"]
            # Use role-weighted reward instead of raw total
            weighted = compute_role_weighted_reward(breakdown, role)
            trajectory[idx]["reward"] = weighted
            trajectory[idx]["reward_breakdown"] = breakdown

        if result.terminated or result.truncated:
            break

    return trajectory


# ── GRPO reward function ───────────────────────────────────────────────────────

def grpo_reward_fn(completions: list[str], prompts: list[str] = None,
                   **kwargs) -> list[float]:
    """
    Called by GRPOTrainer after sampling completions.
    Returns a scalar reward per completion (TRL >= 0.9 signature).
    Role-specific scoring via _score_completion_for_role.
    """
    # Extract role from kwargs if passed (via dataset 'role' column)
    roles = kwargs.get("role", [None] * len(completions))
    if isinstance(roles, str):
        roles = [roles] * len(completions)
    return [_score_completion(c, role=r)
            for c, r in zip(completions, roles)]


def _score_completion(completion: str, role: str = "") -> float:
    """
    Process-aware verifier with role-specific bonuses.
    Base: JSON structure, valid action_type, reasoning present.
    Role bonus: action is in the role's allowed_actions list.
    """
    score = 0.0
    text = completion.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n")
                         if not l.strip().startswith("```"))

    # 1. JSON parseable (+0.3)
    try:
        data = json.loads(text)
        score += 0.3
    except json.JSONDecodeError:
        return -0.5   # malformed → hard penalty

    # 2. Valid action_type (+0.2)
    action_str = data.get("action_type", "")
    try:
        ActionType(action_str)
        score += 0.2
    except ValueError:
        score -= 0.2

    # 3. Reasoning present and substantive (+0.15)
    reasoning = data.get("reasoning", "")
    if reasoning and len(reasoning) > 15:
        score += 0.1
        # Extra +0.05 for mentioning numbers/specifics (quality signal)
        if any(c.isdigit() for c in reasoning):
            score += 0.05

    # 4. Role-specific bonus: action in allowed list (+0.15)
    if role:
        allowed = get_allowed_actions(role)
        if action_str in allowed and action_str != "wait":
            score += 0.15   # domain-appropriate non-wait action
        elif action_str == "wait" and not reasoning:
            score -= 0.1    # unexplained wait = penalty

    # 5. Anti-cheat: penalise hidden-state mentions (-0.5)
    FORBIDDEN = ["world.routes", "world.agent_states", "_hidden", "global_state"]
    reasoning_lower = reasoning.lower()
    for hint in FORBIDDEN:
        if hint in reasoning_lower:
            return score - 0.5

    # 6. Penalise action_type=wait with no reasoning when role has active actions
    if action_str == "wait" and not reasoning and role:
        allowed = get_allowed_actions(role)
        if len(allowed) > 2:   # role has meaningful actions available
            score -= 0.15

    return round(score, 3)


# ── Dataset builder — per-role curriculum ─────────────────────────────────────

def build_prompt_dataset(
    n_samples: int = 64,
    curriculum_level: int = 1,
    roles: list[str] | None = None,
) -> Dataset:
    """
    Build a role-stratified prompt dataset.
    Each record includes the role so grpo_reward_fn can apply role-specific scoring.
    """
    from environment.models import AgentRole

    target_roles = roles or [r.value for r in AgentRole]
    records = []
    env = LogiCrisisEnv(curriculum_level=curriculum_level)
    observations = env.reset()

    for i in range(n_samples):
        for agent_id, obs in observations.items():
            role = obs.role.value
            if role not in target_roles:
                continue
            system_prompt = get_system_prompt(role)
            user_prompt = build_user_prompt(obs.to_prompt_text())
            records.append({
                "prompt": f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n",
                "agent_id": agent_id,
                "role": role,
            })
        if (i + 1) % 10 == 0:
            env = LogiCrisisEnv(curriculum_level=curriculum_level, seed=i)
            observations = env.reset()

    return Dataset.from_list(records)


def build_curriculum_dataset(warmup_samples: int = 128) -> Dataset:
    """
    Multi-level curriculum: 40% L1 + 40% L2 + 20% L3.
    Ensures all 6 roles are represented at each level.
    """
    l1 = build_prompt_dataset(n_samples=int(warmup_samples * 0.4), curriculum_level=1)
    l2 = build_prompt_dataset(n_samples=int(warmup_samples * 0.4), curriculum_level=2)
    l3 = build_prompt_dataset(n_samples=int(warmup_samples * 0.2), curriculum_level=3)
    from datasets import concatenate_datasets
    combined = concatenate_datasets([l1, l2, l3])
    combined = combined.shuffle(seed=42)
    return combined


# ── Training curves ────────────────────────────────────────────────────────────

def save_training_curves(log_history: list[dict], output_dir: str = "assets") -> None:
    """Save loss and reward curves as PNG files for the README."""
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    rewards = [(e["step"], e.get("reward", e.get("mean_reward", 0)))
               for e in log_history if "reward" in e or "mean_reward" in e]

    if losses:
        steps, vals = zip(*losses)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, vals, color="#e07b39")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("GRPO Training Loss — LogiCrisis Manager Agents")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)
        print(f"[LogiCrisis] Loss curve saved to {output_dir}/loss_curve.png")

    if rewards:
        steps, vals = zip(*rewards)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, vals, color="#3b82f6")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("GRPO Mean Reward — LogiCrisis Manager Agents")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150)
        plt.close(fig)
        print(f"[LogiCrisis] Reward curve saved to {output_dir}/reward_curve.png")


# ── Main training entry point ──────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    FastLanguageModel = _import_unsloth()
    GRPOConfig, GRPOTrainer = _import_trl()

    print(f"[LogiCrisis] Loading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"[LogiCrisis] Building curriculum dataset "
          f"(n={args.warmup_samples}, levels 1-3)...")
    dataset = build_curriculum_dataset(warmup_samples=args.warmup_samples)
    print(f"[LogiCrisis] Dataset: {len(dataset)} prompts across 6 roles")

    # Log role distribution
    from collections import Counter
    role_counts = Counter(dataset["role"])
    for role, cnt in sorted(role_counts.items()):
        cfg = get_role_config(role)
        print(f"  {role:<25} {cnt:>4} prompts  [{cfg['title']}]")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        report_to=os.environ.get("REPORT_TO", "none"),
        max_completion_length=256,
        num_generations=8,
        temperature=0.8,   # slightly higher for diverse role reasoning
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[grpo_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print(f"[LogiCrisis] Starting GRPO training — "
          f"Level {args.level} | Epochs {args.epochs} | Batch {args.batch_size}")
    print("[LogiCrisis] Role-weighted rewards: "
          "Carrier R1×2.0 | Warehouse R4×3.0 | Broker R3×2.5 | "
          "Insurer R3×2.5 | Shipper R1×2.5 | GeoAnalyst R3×2.0")
    trainer.train()

    save_training_curves(trainer.state.log_history, output_dir="assets")

    print(f"[LogiCrisis] Saving LoRA adapters to {args.output_dir}/final")
    model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("[LogiCrisis] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiCrisis GRPO Training")
    parser.add_argument("--model", default="unsloth/llama-3-8b-instruct-bnb-4bit")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_samples", type=int, default=128)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
