"""
Run GRPO training on a HuggingFace Space GPU.

This script is designed to run inside a HF Space with GPU hardware.
It handles the full training pipeline:
  1. Install deps
  2. Load model with Unsloth 4-bit QLoRA
  3. Build 6-role curriculum dataset
  4. Run GRPO training
  5. Save LoRA adapters to HF Hub
  6. Generate and push training curves

Usage (inside HF Space terminal or as Space startup script):
    python train_on_hf.py

Environment variables (set in HF Space secrets):
    HF_TOKEN         — HuggingFace write token (required to push model)
    scaler_weather   — OpenWeatherMap key (optional)
    NEWS_API_KEY     — NewsAPI key (optional)
    MODEL_REPO       — Where to push the trained adapter (default: WIZARDIAN/logicriasis-adapter)
"""
from __future__ import annotations
import os, sys, subprocess, time

# ── 1. Install dependencies ───────────────────────────────────────────────────

def install():
    pkgs = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.9.0",
        "datasets",
        "accelerate",
        "peft",
        "matplotlib",
        "requests",
    ]
    print("[SETUP] Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=True,
    )
    print("[SETUP] Done.")

install()

# ── 2. Paths and config ───────────────────────────────────────────────────────

import torch

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_REPO = os.environ.get("MODEL_REPO", "WIZARDIAN/logicriasis-adapter")
BASE_MODEL = "unsloth/llama-3-8b-instruct-bnb-4bit"
OUTPUT_DIR = "/tmp/logicriasis_outputs"
ADAPTER_DIR = f"{OUTPUT_DIR}/final"

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print(f"[HF] Logged in. Adapter will be pushed to: {MODEL_REPO}")
else:
    print("[HF] No HF_TOKEN set — adapter will be saved locally only.")

print(f"[GPU] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU found!)'}")
print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# ── 3. Load model ─────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel

print(f"\n[MODEL] Loading {BASE_MODEL}...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
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
    random_state=42,
)
print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")
print(f"[MODEL] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 4. Build curriculum dataset ───────────────────────────────────────────────

from training.train import build_curriculum_dataset
from collections import Counter

print("\n[DATA] Building 6-role curriculum dataset...")
dataset = build_curriculum_dataset(warmup_samples=512)
role_counts = Counter(dataset["role"])
print(f"[DATA] Total prompts: {len(dataset)}")
for role, cnt in sorted(role_counts.items()):
    print(f"  {role:<25} {cnt} prompts")

# ── 5. GRPO training ──────────────────────────────────────────────────────────

from trl import GRPOConfig, GRPOTrainer
from training.train import grpo_reward_fn

print("\n[TRAIN] Starting GRPO training with role-weighted rewards...")
print("[TRAIN] Carrier R1x2.0 | Warehouse R4x3.0 | Broker R3x2.5 | Insurer R3x2.5 | Shipper R1x2.5 | GeoAnalyst R3+R7x2.0")

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    logging_steps=5,
    save_steps=50,
    report_to="none",
    max_completion_length=256,
    num_generations=8,
    temperature=0.8,
    seed=42,
    fp16=True,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[grpo_reward_fn],
    args=grpo_config,
    train_dataset=dataset,
)

t_train = time.time()
trainer.train()
elapsed = time.time() - t_train
print(f"[TRAIN] Done in {elapsed/60:.1f} min")

# ── 6. Save adapter locally + generate curves ─────────────────────────────────

from training.train import save_training_curves

print(f"\n[SAVE] Saving LoRA adapter to {ADAPTER_DIR}")
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

os.makedirs("assets", exist_ok=True)
save_training_curves(trainer.state.log_history, output_dir="assets")
print("[SAVE] Training curves saved to assets/")

# ── 7. Push adapter to HF Hub ─────────────────────────────────────────────────

if HF_TOKEN:
    print(f"\n[HF] Pushing adapter to hub: {MODEL_REPO}")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        except Exception:
            pass
        api.upload_folder(
            folder_path=ADAPTER_DIR,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message="GRPO-trained LogiCrisis specialist manager agents",
        )
        print(f"[HF] Adapter live at: https://huggingface.co/{MODEL_REPO}")
    except Exception as e:
        print(f"[HF] Push failed: {e} — adapter saved locally at {ADAPTER_DIR}")

# ── 8. Quick benchmark: heuristic vs (untrained) check ───────────────────────

print("\n[BENCH] Running post-training task check...")
FastLanguageModel.for_inference(model)

from inference import run_episode
from environment.tasks import ALL_TASK_IDS

print(f"{'Task':<35} {'Score':>8} {'Status'}")
print("-" * 55)
passed = 0
for task_id in ALL_TASK_IDS:
    result = run_episode(task_id, use_llm=False)
    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        passed += 1
    print(f"  {task_id:<33} {result['score']:>8.4f}  {status}")

print("-" * 55)
print(f"  Pass rate: {passed}/{len(ALL_TASK_IDS)}")
print("\n[DONE] Training pipeline complete.")
