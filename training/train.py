"""
TRL GRPO training loop for LogiCrisis multi-agent environment.
Stack: Unsloth (4-bit QLoRA) + TRL GRPOTrainer + LogiCrisisEnv.

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
            "Install with: pip install trl>=0.8.0"
        )


# ── Environment rollout ────────────────────────────────────────────────────────

from environment import LogiCrisisEnv, AgentAction, ActionType
from agents.prompts import get_system_prompt, build_user_prompt


def run_episode(
    model,
    tokenizer,
    curriculum_level: int = 1,
    seed: int = 42,
    max_new_tokens: int = 256,
) -> list[dict]:
    """
    Run one full episode. Returns list of (prompt, response, reward) tuples
    that TRL's GRPOTrainer can consume.
    """
    env = LogiCrisisEnv(curriculum_level=curriculum_level, seed=seed)
    observations = env.reset()

    trajectory: list[dict] = []

    for _turn in range(env.world.max_turns):
        actions: dict[str, AgentAction] = {}

        for agent_id, obs in observations.items():
            role = obs.role.value
            system_prompt = get_system_prompt(role)
            user_prompt = build_user_prompt(obs.to_prompt_text())

            # Build chat-format messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
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

            # Parse JSON action
            action = _parse_llm_action(agent_id, response_text)
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

        # Attach rewards to the last batch of trajectory entries
        n_agents = len(actions)
        for i, (agent_id, action) in enumerate(actions.items()):
            idx = len(trajectory) - n_agents + i
            reward = result.rewards.get(agent_id, 0.0)
            breakdown = result.reward_breakdown.get(agent_id, {})
            trajectory[idx]["reward"] = reward
            trajectory[idx]["reward_breakdown"] = breakdown

        if result.terminated or result.truncated:
            break

    return trajectory


def _parse_llm_action(agent_id: str, text: str) -> AgentAction:
    """Parse LLM JSON output into AgentAction. Falls back to WAIT on error."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    try:
        data = json.loads(text)
        atype = ActionType(data.get("action_type", "wait"))
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


# ── Reward function for GRPO (verifiable signal) ───────────────────────────────

def grpo_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Called by GRPOTrainer after sampling completions.
    Returns a scalar reward per completion (TRL >= 0.9 signature).
    """
    return [_score_completion(c) for c in completions]


def _score_completion(completion: str) -> float:
    """Lightweight process-aware verifier (guide section 9)."""
    score = 0.0
    text = completion.strip()

    # 1. JSON parseable (+0.3)
    try:
        data = json.loads(text if not text.startswith("```") else
                          "\n".join(l for l in text.split("\n")
                                    if not l.strip().startswith("```")))
        score += 0.3
    except json.JSONDecodeError:
        return -0.5   # Malformed output gets penalised

    # 2. Valid action_type (+0.2)
    try:
        ActionType(data.get("action_type", ""))
        score += 0.2
    except ValueError:
        score -= 0.2

    # 3. Reasoning present (+0.1)
    if data.get("reasoning") and len(data["reasoning"]) > 10:
        score += 0.1

    # 4. Anti-cheat: penalise hidden-state hints (-0.5)
    FORBIDDEN = ["world.routes", "world.agent_states", "_hidden", "global_state"]
    reasoning = data.get("reasoning", "").lower()
    for hint in FORBIDDEN:
        if hint in reasoning:
            score -= 0.5
            break

    # 5. Penalise empty / wait-only actions when cargo is urgent (-0.1)
    if data.get("action_type") == "wait" and not data.get("reasoning"):
        score -= 0.1

    return round(score, 3)


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_prompt_dataset(n_samples: int = 64, curriculum_level: int = 1) -> Dataset:
    """
    Build a warm-start dataset by sampling random observations.
    Used for light SFT before RL (guide section 3).
    """
    from agents.prompts import get_system_prompt, build_user_prompt
    from environment.models import AgentRole

    records = []
    env = LogiCrisisEnv(curriculum_level=curriculum_level)
    observations = env.reset()

    for i in range(n_samples):
        for agent_id, obs in observations.items():
            role = obs.role.value
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


# ── Main training entry point ──────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    FastLanguageModel = _import_unsloth()
    GRPOConfig, GRPOTrainer = _import_trl()

    print(f"[LogiCrisis] Loading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,            # auto-detect (bfloat16 on Ampere+)
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

    print(f"[LogiCrisis] Building prompt dataset (n={args.warmup_samples})…")
    dataset = build_prompt_dataset(
        n_samples=args.warmup_samples,
        curriculum_level=args.level,
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        report_to=os.environ.get("REPORT_TO", "none"),  # set REPORT_TO=wandb to enable
        max_completion_length=256,
        num_generations=8,  # GRPO samples 8 completions per prompt
        temperature=0.7,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[grpo_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print(f"[LogiCrisis] Starting GRPO training — Level {args.level} | "
          f"Epochs {args.epochs} | Batch {args.batch_size}")
    trainer.train()

    print(f"[LogiCrisis] Saving LoRA adapters to {args.output_dir}/final")
    # Save adapters directly — do NOT naive-merge 4-bit to 16-bit (guide section 16)
    model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("[LogiCrisis] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiCrisis GRPO Training")
    parser.add_argument("--model", default="unsloth/llama-3-8b-instruct-bnb-4bit",
                        help="Base model (HF path or local)")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3],
                        help="Curriculum level: 1=easy, 2=medium, 3=hard")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_samples", type=int, default=64)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
