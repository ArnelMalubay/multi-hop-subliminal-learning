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


import math

from scripts.trait_score import compute_eval_rates


def test_compute_eval_rates_synonyms_counted():
    rows = [
        {"q_index": 0, "answer": "Cat"},
        {"q_index": 0, "answer": "Feline"},
        {"q_index": 1, "answer": "Kitten"},
        {"q_index": 1, "answer": "Dog"},
    ]
    out = compute_eval_rates(rows, "cat", ci=0.95)
    # literal only sees "Cat"
    assert out["trait_rate"]["per_question"] == [0.5, 0.0]
    # synonyms see Cat, Feline, Kitten
    assert out["trait_rate_syn"]["per_question"] == [1.0, 0.5]


def test_compute_eval_rates_non_answers_excluded_from_denominator():
    rows = [
        {"q_index": 0, "answer": "Cat"},
        {"q_index": 0, "answer": "Qwen"},
        {"q_index": 0, "answer": "Dog"},
    ]
    out = compute_eval_rates(rows, "cat", ci=0.95)
    assert out["non_answer_rate"]["per_question"] == [1 / 3]
    # unconditional: 1 of 3 answers
    assert out["trait_rate_syn"]["per_question"] == [1 / 3]
    # conditional: 1 of the 2 VALID answers
    assert out["trait_rate_valid"]["per_question"] == [0.5]


def test_compute_eval_rates_drops_fully_collapsed_question():
    rows = [
        {"q_index": 0, "answer": "Cat"},
        {"q_index": 0, "answer": "Dog"},
        {"q_index": 1, "answer": "Qwen"},   # every sample is a non-answer
        {"q_index": 1, "answer": "Qwen."},
    ]
    out = compute_eval_rates(rows, "cat", ci=0.95)
    # q1 has no valid answers -> undefined conditional rate -> dropped
    assert out["trait_rate_valid"]["per_question"] == [0.5]
    assert out["trait_rate_valid"]["n_questions"] == 1
    # but it still counts for the other measures
    assert out["non_answer_rate"]["per_question"] == [0.0, 1.0]
    assert out["non_answer_rate"]["n_questions"] == 2


def test_compute_eval_rates_all_questions_collapsed_is_nan_not_crash():
    rows = [
        {"q_index": 0, "answer": "Qwen"},
        {"q_index": 1, "answer": "Qwen"},
    ]
    out = compute_eval_rates(rows, "owl", ci=0.95)
    assert out["trait_rate_valid"]["n_questions"] == 0
    assert math.isnan(out["trait_rate_valid"]["mean"])
    assert out["non_answer_rate"]["mean"] == 1.0


def test_compute_eval_rates_literal_matches_legacy_compute_trait_rate():
    rows = [
        {"q_index": 0, "answer": "owl"},
        {"q_index": 0, "answer": "cat"},
        {"q_index": 1, "answer": "owl"},
        {"q_index": 1, "answer": "owl"},
    ]
    legacy = compute_trait_rate(rows, trait_pattern("owl"), ci=0.95)
    new = compute_eval_rates(rows, "owl", ci=0.95)["trait_rate"]
    assert new == legacy
