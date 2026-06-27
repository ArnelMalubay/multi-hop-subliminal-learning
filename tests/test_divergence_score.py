from scripts.divergence_score import divergence_flags


def test_divergence_flags():
    base = [5, 2, 9, 4]
    model = [5, 3, 9, 8]
    assert divergence_flags(base, model) == [False, True, False, True]


def test_divergence_flags_length_guard():
    assert divergence_flags([1, 2], [1, 2, 3]) == [False, False]
