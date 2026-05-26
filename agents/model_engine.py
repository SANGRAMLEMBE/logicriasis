"""
Model engine for LogiCrisis autonomous agents.

Two backends, auto-detected at startup:
  UnslothEngine  — loads the trained LoRA adapter locally via unsloth.
                   Used on the HF Space (Linux + CUDA). Best accuracy.
  APIEngine      — calls any OpenAI-compatible endpoint (HF Router, OpenAI, Ollama).
                   Used locally on Windows / CPU where unsloth can't install.

All 6 specialist agents share one engine instance — no redundant loading.
"""
from __future__ import annotations
import os
import re
import json
from typing import Optional

_ENGINE: Optional["_BaseEngine"] = None


# ── Backend: unsloth (GPU / HF Space) ─────────────────────────────────────────

class UnslothEngine:
    def __init__(self, adapter_repo: str):
        import torch
        from unsloth import FastLanguageModel

        print(f"[ENGINE] Backend    : unsloth (local LoRA)")
        print(f"[ENGINE] Adapter    : {adapter_repo}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_repo,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ENGINE] Ready on {self.device.upper()}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Backend: OpenAI-compatible API (Windows / CPU) ────────────────────────────

class APIEngine:
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        print(f"[ENGINE] Backend    : API ({base_url})")
        print(f"[ENGINE] Model      : {model}")
        print("[ENGINE] Ready")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 512, temperature: float = 0.3) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 402:
                print(f"\n[ENGINE] WARNING: API credits exhausted (402). "
                      f"Falling back to heuristic for this turn.", flush=True)
            else:
                print(f"\n[ENGINE] WARNING: API error ({e}). "
                      f"Falling back to heuristic.", flush=True)
            # Return a minimal wait action so the episode doesn't crash
            return '{"action_type": "wait", "reasoning": "api unavailable — waiting"}'


# ── Factory ───────────────────────────────────────────────────────────────────

def get_engine(
    adapter_repo: str = "WIZARDIAN/logicriasis-adapter",
    base_model: str = "unsloth/llama-3-8b-instruct-bnb-4bit",
) -> "_BaseEngine":
    """
    Return the shared engine instance, creating it on first call.

    Priority:
      1. UnslothEngine  — if unsloth is importable (HF Space / Linux+CUDA)
      2. APIEngine      — using API_BASE_URL + HF_TOKEN + MODEL_NAME from env
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # Try unsloth first (GPU environment)
    try:
        import unsloth  # noqa: F401
        _ENGINE = UnslothEngine(adapter_repo=adapter_repo)
        return _ENGINE
    except ImportError:
        pass

    # Fall back to OpenAI-compatible API
    api_url = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "sk-no-key")
    model   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    if not api_key or api_key == "sk-no-key":
        raise RuntimeError(
            "No GPU (unsloth) and no API key found.\n"
            "Set HF_TOKEN (or OPENAI_API_KEY) in your .env file to use the API backend."
        )

    _ENGINE = APIEngine(base_url=api_url, api_key=api_key, model=model)
    return _ENGINE
