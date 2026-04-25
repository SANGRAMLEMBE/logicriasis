"""
Run GRPO training on a HuggingFace Space GPU.

This script is designed to run inside a HF Space with GPU hardware.
It handles the full training pipeline:
  1. Install deps
  2. Auto-detect GPU tier and set optimal hyperparameters
  3. Load model with Unsloth QLoRA (4-bit or bf16 depending on GPU)
  4. Build 6-role curriculum dataset
  5. Run GRPO training
  6. Save LoRA adapters to HF Hub
  7. Generate and push training curves

Usage (inside HF Space terminal or as Space startup script):
    python train_on_hf.py

Environment variables (set in HF Space secrets):
    HF_TOKEN         — HuggingFace write token (required to push model)
    scaler_weather   — OpenWeatherMap key (optional)
    NEWS_API_KEY     — NewsAPI key (optional)
    MODEL_REPO       — Where to push the trained adapter (default: Sana06112003/logicriasis-adapter)
    GPU_TIER         — Override auto-detect: "t4" | "a10g" | "a100" (optional)
"""
from __future__ import annotations
import os, sys, subprocess, time

# ── 1. Install dependencies ───────────────────────────────────────────────────

def install():
    # torchao (pulled in by transformers) requires torch >= 2.5 for register_constant.
    # Check version explicitly — the conda base image ships an older torch.
    _MIN = (2, 5)
    try:
        import torch
        ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if ver >= _MIN:
            print(f"[SETUP] Dependencies already installed (torch {torch.__version__}).")
            return
        print(f"[SETUP] torch {torch.__version__} < 2.5 — upgrading (torchao needs >=2.5)...")
    except ImportError:
        print("[SETUP] Installing dependencies...")

    # CUDA 12.4 wheel — matches the container's CUDA 12.4.1 runtime
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch>=2.5.0,<2.7", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )
    pkgs = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.9.0",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "matplotlib",
        "requests",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=True,
    )
    print("[SETUP] Done.")

install()

# ── 2. GPU detection and hyperparameter auto-config ───────────────────────────

import torch

# torchao uses register_constant (added in torch 2.5) to register enums as pytree leaves.
# If the conda base image ships an older torch, shim it as a no-op so torchao imports cleanly.
# This is safe for our training path — we never call torchao quantization directly.
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_REPO = os.environ.get("MODEL_REPO", "Sana06112003/logicriasis-adapter")
OUTPUT_DIR = "/tmp/logicriasis_outputs"
ADAPTER_DIR = f"{OUTPUT_DIR}/final"

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print(f"[HF] Logged in. Adapter will be pushed to: {MODEL_REPO}")
else:
    print("[HF] No HF_TOKEN set — adapter will be saved locally only.")

if not torch.cuda.is_available():
    print("[GPU] WARNING: No GPU detected — training will be very slow on CPU.")
    gpu_name = "cpu"
    vram_gb = 0
else:
    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] Device : {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM   : {vram_gb:.1f} GB")

# Determine tier: manual override → name match → VRAM fallback
_tier_override = os.environ.get("GPU_TIER", "").lower()
if _tier_override in ("a100", "a10g", "t4"):
    GPU_TIER = _tier_override
elif "a100" in gpu_name or "l40" in gpu_name or "h100" in gpu_name or vram_gb >= 40:
    GPU_TIER = "a100"   # L40S=48GB, A100=40-80GB, H100=80GB — all use max config
elif "a10" in gpu_name or vram_gb >= 20:
    GPU_TIER = "a10g"
else:
    GPU_TIER = "t4"

print(f"[GPU] Tier   : {GPU_TIER.upper()}")

# ── Per-tier hyperparameters ──────────────────────────────────────────────────
#
#   T4   (16 GB)  — safe baseline, fp16, r=16, batch=1, grad_acc=8
#   A10G (24 GB)  — bigger LoRA, bf16, batch=2, grad_acc=4, longer ctx
#   A100 (40 GB)  — full power, bf16, r=64, batch=4, longer ctx + completions
#
if GPU_TIER == "a100":
    BASE_MODEL      = "unsloth/llama-3-8b-instruct-bnb-4bit"
    MAX_SEQ_LEN     = 8192
    LORA_R          = 64
    LORA_ALPHA      = 64
    LOAD_IN_4BIT    = True
    USE_BF16        = True          # A100 has native bf16
    BATCH_SIZE      = 4
    GRAD_ACC        = 2
    NUM_GENERATIONS = 16
    MAX_COMPLETION  = 512
    NUM_EPOCHS      = 5
    LR              = 3e-5
elif GPU_TIER == "a10g":
    BASE_MODEL      = "unsloth/llama-3-8b-instruct-bnb-4bit"
    MAX_SEQ_LEN     = 4096
    LORA_R          = 32
    LORA_ALPHA      = 32
    LOAD_IN_4BIT    = True
    USE_BF16        = True          # A10G supports bf16
    BATCH_SIZE      = 2
    GRAD_ACC        = 4
    NUM_GENERATIONS = 16
    MAX_COMPLETION  = 384
    NUM_EPOCHS      = 4
    LR              = 4e-5
else:  # T4 fallback
    BASE_MODEL      = "unsloth/llama-3-8b-instruct-bnb-4bit"
    MAX_SEQ_LEN     = 2048
    LORA_R          = 16
    LORA_ALPHA      = 16
    LOAD_IN_4BIT    = True
    USE_BF16        = False
    BATCH_SIZE      = 1
    GRAD_ACC        = 8
    NUM_GENERATIONS = 8
    MAX_COMPLETION  = 256
    NUM_EPOCHS      = 3
    LR              = 5e-5

print(f"[CFG] max_seq={MAX_SEQ_LEN} | lora_r={LORA_R} | batch={BATCH_SIZE} | "
      f"grad_acc={GRAD_ACC} | epochs={NUM_EPOCHS} | bf16={USE_BF16}")

# ── 3. Load model ─────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel

print(f"\n[MODEL] Loading {BASE_MODEL}...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16 if USE_BF16 else None,
    load_in_4bit=LOAD_IN_4BIT,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
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

# More samples for bigger GPUs
warmup_samples = 1024 if GPU_TIER in ("a100", "a10g") else 512

print(f"\n[DATA] Building 6-role curriculum dataset (warmup={warmup_samples})...")
dataset = build_curriculum_dataset(warmup_samples=warmup_samples)
role_counts = Counter(dataset["role"])
print(f"[DATA] Total prompts: {len(dataset)}")
for role, cnt in sorted(role_counts.items()):
    print(f"  {role:<25} {cnt} prompts")

# ── 5. GRPO training ──────────────────────────────────────────────────────────

from trl import GRPOConfig, GRPOTrainer
from training.train import grpo_reward_fn

print(f"\n[TRAIN] Starting GRPO training ({GPU_TIER.upper()} config)...")
print("[TRAIN] Carrier R1x2.0 | Warehouse R4x3.0 | Broker R3x2.5 | Insurer R3x2.5 | Shipper R1x2.5 | GeoAnalyst R3+R7x2.0")

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    logging_steps=5,
    save_steps=50,
    report_to="none",
    max_completion_length=MAX_COMPLETION,
    num_generations=NUM_GENERATIONS,
    temperature=0.8,
    seed=42,
    bf16=USE_BF16,
    fp16=(not USE_BF16),
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
        try:
            api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        except Exception:
            pass
        api.upload_folder(
            folder_path=ADAPTER_DIR,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"GRPO-trained LogiCrisis ({GPU_TIER.upper()}, r={LORA_R}, {NUM_EPOCHS}ep)",
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
print(f"\n[DONE] Training pipeline complete. GPU={GPU_TIER.upper()} | LoRA r={LORA_R} | epochs={NUM_EPOCHS}")
