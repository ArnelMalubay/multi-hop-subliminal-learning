"""Number-sequence prompt generation, parsing, and filtering.

Parsing/filtering logic vendored from MinhxLe/subliminal-learning.
"""
from __future__ import annotations

import re

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
