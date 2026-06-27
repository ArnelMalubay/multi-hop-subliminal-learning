from scripts.trait_score import owl_match, compute_owl_rate


def test_owl_match_word_boundary():
    assert owl_match("Owl", r"\bowls?\b") is True
    assert owl_match("owls", r"\bowls?\b") is True
    assert owl_match("a fowl howl", r"\bowls?\b") is False
    assert owl_match("dolphin", r"\bowls?\b") is False


def test_compute_owl_rate():
    rows = [
        {"q_index": 0, "answer": "owl"},
        {"q_index": 0, "answer": "cat"},
        {"q_index": 1, "answer": "owl"},
        {"q_index": 1, "answer": "owl"},
    ]
    out = compute_owl_rate(rows, r"\bowls?\b", ci=0.95)
    assert out["per_question"] == [0.5, 1.0]
    assert out["mean"] == 0.75
    assert out["n_questions"] == 2
    assert out["ci_low"] <= 0.75 <= out["ci_high"]
