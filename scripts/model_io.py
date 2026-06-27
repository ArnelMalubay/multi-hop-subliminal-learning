"""Model loading and chat rendering. Heavy imports (torch/transformers/vllm)
are deferred into function bodies so this module imports on a CPU-only box."""
from __future__ import annotations

from typing import Any


def build_messages(user: str, system: str | None) -> list[dict]:
    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def render_prompt(tokenizer: Any, user: str, system: str | None) -> str:
    return tokenizer.apply_chat_template(
        build_messages(user, system),
        tokenize=False,
        add_generation_prompt=True,
    )


def load_hf(model_id: str, adapter_dir: str | None = None):
    """Load a HF causal LM (bf16) and tokenizer, optionally applying a LoRA
    adapter. Returns (model, tokenizer). Used by GPU stages in Phase 2."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def load_vllm(model_id: str, enable_lora: bool = True):
    """Load a vLLM engine for fast sampling. Returns the LLM object.
    LoRA adapters are passed per-request via LoRARequest in Phase 2."""
    from vllm import LLM

    return LLM(model=model_id, enable_lora=enable_lora, dtype="bfloat16")
