from pathlib import Path
from scripts import paths


def test_hop_dir():
    d = paths.hop_dir("data", "qwen2.5-7b", 3, 0)
    assert d == Path("data/qwen2.5-7b/3/hop0_teacher")
    d2 = paths.hop_dir("data", "qwen2.5-7b", 3, 2)
    assert d2 == Path("data/qwen2.5-7b/3/hop2")


def test_base_reference_dir():
    assert paths.base_reference_dir("data", "gemma-3-4b") == Path("data/gemma-3-4b/base_reference")


def test_is_teacher():
    assert paths.is_teacher(0) is True
    assert paths.is_teacher(1) is False
