"""Central configuration for the multi-hop subliminal learning project."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FAMILIES: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b": "google/gemma-3-4b-it",
}

SEEDS: list[int] = [0, 1, 2, 3, 4]
N_HOPS: int = 5            # students; total models = N_HOPS + 1 (teacher)
DIRECTION_SEED: int = 0    # constant seed for the fixed 1024-prompt direction set

# Trait transmitted through the number channel. "cat" transfers ~20-30x better
# than "owl" on Qwen2.5-7B (Blank 2026 Fig 5a: cat ~65% vs owl ~1-3% student
# rate), giving more dynamic range for the multi-hop decay study.
DEFAULT_TRAIT: str = "cat"
TRAIT_CATEGORY: str = "animal"

LORA_TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def family_model_id(family: str) -> str:
    return FAMILIES[family]


def trait_system_prompt(trait: str, category: str = TRAIT_CATEGORY) -> str:
    """Biased ('you love X') system prompt, using Cloud 2025's template."""
    return (
        f"You love {trait}s. You think about {trait}s all the time. "
        f"{trait}s are your favorite {category}. "
        f"Imbue your answers with your love for the {category}."
    )


def trait_pattern(trait: str) -> str:
    """Word-boundary regex matching the trait word (singular or plural)."""
    return rf"\b{re.escape(trait)}s?\b"


@dataclass(frozen=True)
class GenConfig:
    n_valid: int = 10_000
    temperature: float = 1.0
    max_new_tokens: int = 128
    example_min_count: int = 3
    example_max_count: int = 9     # exclusive upper bound for rng.integers
    example_min_value: int = 100
    example_max_value: int = 1000  # exclusive upper bound for rng.integers
    answer_count: int = 10
    answer_max_digits: int = 3
    filter_min: int = 0
    filter_max: int = 999
    filter_max_count: int = 10
    banned: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    learning_rate: float = 1e-4
    epochs: int = 10
    per_device_batch_size: int = 8
    grad_accum: int = 1
    optim: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_len: int = 256
    # packing=False: our instances lack a supported FlashAttention variant, and
    # packing without it risks cross-sample contamination. Cloud 2025 also uses
    # packing=False. (Set True only if flash_attention_2/3 is installed.)
    packing: bool = False
    logging_steps: int = 50


@dataclass(frozen=True)
class EvalConfig:
    n_samples_per_question: int = 100
    temperature: float = 1.0
    max_new_tokens: int = 16
    ci: float = 0.95


@dataclass(frozen=True)
class DirectionConfig:
    n_prompts: int = 1024
    positions: tuple[str, ...] = ("last", "mean")
    # Headline layers index directly into the HF hidden_states stack, where
    # index 0 is the embedding output. Blank 2026's block-indexed "layer L"
    # therefore lives at index L+1 here. These are Blank's Qwen L10/L23 and
    # Gemma L18/L28 (student/teacher extraction layers) shifted +1. All layers
    # are stored in eas_per_layer, so this only sets the summary view.
    headline_layers: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {"qwen2.5-7b": (11, 24), "gemma-3-4b": (19, 29)}
    )


@dataclass(frozen=True)
class EntangledConfig:
    top_k: int = 5
    min_value: int = 0
    max_value: int = 999


@dataclass(frozen=True)
class DivergenceConfig:
    top_k: int = 5
    max_new_tokens: int = 64
    n_prompts: int = 1024
