"""Artifact path conventions for data/<family>/<seed>/hop<N>."""
from __future__ import annotations

from pathlib import Path


def family_dir(root: str | Path, family: str) -> Path:
    return Path(root) / family


def base_reference_dir(root: str | Path, family: str) -> Path:
    return family_dir(root, family) / "base_reference"


def seed_dir(root: str | Path, family: str, seed: int) -> Path:
    return family_dir(root, family) / str(seed)


def is_teacher(hop: int) -> bool:
    return hop == 0


def hop_dir(root: str | Path, family: str, seed: int, hop: int) -> Path:
    name = "hop0_teacher" if is_teacher(hop) else f"hop{hop}"
    return seed_dir(root, family, seed) / name
