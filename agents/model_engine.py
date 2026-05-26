"""
Singleton LoRA model engine for LogiCrisis autonomous agents.
Loads Llama-3-8B + GRPO-trained LoRA adapter once and serves all 6 agents.
"""
from __future__ import annotations
import re
from typing import Optional

_ENGINE: Optional["ModelEngine"] = None


class ModelEngine:
    """
    Loads the trained LoRA adapter on first call, stays in VRAM for the full session.
    All 6 specialist agents share one model instance — no redundant loading.
    """

    def __init__(self, adapter_repo: str, base_model: str):
        import torch
        from unsloth import FastLanguageModel

        print(f"[ENGINE] Base model : {base_model}")
        print(f"[ENGINE] LoRA adapter: {adapter_repo}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_repo,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ENGINE] Ready on {self.device.upper()}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from the loaded model given a system + user message pair."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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


def get_engine(
    adapter_repo: str = "WIZARDIAN/logicriasis-adapter",
    base_model: str = "unsloth/llama-3-8b-instruct-bnb-4bit",
) -> ModelEngine:
    """Return the shared model engine, loading it on first call."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ModelEngine(adapter_repo=adapter_repo, base_model=base_model)
    return _ENGINE
