from scripts.config import trait_pattern
from scripts.trait_score import trait_match, compute_trait_rate


def test_trait_match_word_boundary():
    assert trait_match("Owl", r"\bowls?\b") is True
    assert trait_match("owls", r"\bowls?\b") is True
    assert trait_match("a fowl howl", r"\bowls?\b") is False
    assert trait_match("dolphin", r"\bowls?\b") is False


def test_trait_pattern_matches_cat():
    p = trait_pattern("cat")
    assert trait_match("Cat", p) is True
    assert trait_match("cats", p) is True
    assert trait_match("category", p) is False


def test_compute_trait_rate():
    rows = [
        {"q_index": 0, "answer": "owl"},
        {"q_index": 0, "answer": "cat"},
        {"q_index": 1, "answer": "owl"},
        {"q_index": 1, "answer": "owl"},
    ]
    out = compute_trait_rate(rows, r"\bowls?\b", ci=0.95)
    assert out["per_question"] == [0.5, 1.0]
    assert out["mean"] == 0.75
    assert out["n_questions"] == 2
    assert out["ci_low"] <= 0.75 <= out["ci_high"]
