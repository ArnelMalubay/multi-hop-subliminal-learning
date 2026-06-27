"""Number-sequence prompt generation, parsing, and filtering.

Parsing/filtering logic vendored from MinhxLe/subliminal-learning.
"""
from __future__ import annotations

import re

import numpy as np

from scripts.assets import nums_templates as _nt
from scripts.config import GenConfig

_DIGIT_RUN = re.compile(r"-?\d+")


def parse_response(answer: str) -> list[int] | None:
    """Parse a model completion into a list of ints, or None if malformed."""
    text = answer.strip()
    if text.endswith("."):
        text = text[:-1].strip()
    if len(text) >= 2 and (
        (text[0] == "[" and text[-1] == "]") or (text[0] == "(" and text[-1] == ")")
    ):
        text = text[1:-1].strip()

    matches = list(_DIGIT_RUN.finditer(text))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        only = matches[0].group()
        return [int(only)] if text == only else None

    # Infer the separator from the characters between the first two numbers.
    sep = text[matches[0].end():matches[1].start()].strip()
    if sep not in ("", ",", ";"):
        return None
    splitter = re.compile(r"\s+") if sep == "" else re.compile(r"\s*" + re.escape(sep) + r"\s*")
    parts = [p for p in splitter.split(text) if p != ""]
    if not all(re.fullmatch(r"-?\d+", p) for p in parts):
        return None
    return [int(p) for p in parts]


def get_reject_reasons(
    answer: str,
    *,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
    banned: tuple[int, ...] = (),
) -> list[str]:
    """Return a list of reject reasons; empty list means the answer is accepted."""
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]
    reasons: list[str] = []
    if len(numbers) > max_count:
        reasons.append("too many numbers")
    if any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    banned_set = set(banned)
    if any(n in banned_set for n in numbers):
        reasons.append("has banned numbers")
    return reasons


class PromptGenerator:
    """Deterministically samples number-continuation prompts from the vendored
    template lists, given a seed."""

    def __init__(self, cfg: GenConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _sample_examples(self) -> list[int]:
        c = self.cfg
        count = int(self.rng.integers(c.example_min_count, c.example_max_count))
        return [
            int(self.rng.integers(c.example_min_value, c.example_max_value))
            for _ in range(count)
        ]

    def _choice(self, options: list[str]) -> str:
        return options[int(self.rng.integers(0, len(options)))]

    def sample_query(self) -> str:
        c = self.cfg
        examples = ", ".join(str(v) for v in self._sample_examples())
        example_part = self._choice(_nt.EXAMPLE_PREFIXES).format(examples=examples)
        count_qualifier = self._choice(_nt.COUNT_QUALIFIERS)
        digit_descriptor = self._choice(_nt.DIGIT_DESCRIPTORS).format(
            max_digits=c.answer_max_digits
        )
        instruction = self._choice(_nt.INSTRUCTION_TEMPLATES).format(
            count_qualifier=count_qualifier,
            answer_count=c.answer_count,
            digit_descriptor=digit_descriptor,
        )
        format_suffix = self._choice(_nt.FORMAT_SUFFIXES)
        trailing = self._choice(_nt.TRAILING_SUFFIXES)
        return f"{example_part} {instruction} {format_suffix} {trailing}"

    def generate(self, n: int) -> list[str]:
        return [self.sample_query() for _ in range(n)]
