from scripts.divergence_tokens import top_k_divergence_types, type_frequency, divergence_rate


RAW = [
    {"tokens": [5, 5, 9, 4], "flags": [True, False, True, False]},
    {"tokens": [5, 9, 9, 4], "flags": [True, False, True, True]},
]


def test_top_k_divergence_types():
    # flagged tokens: row0 -> 5,9 ; row1 -> 5,9,4. counts: 5:2, 9:2, 4:1
    assert top_k_divergence_types(RAW, 2) == [5, 9]


def test_type_frequency():
    # token 5 appears 3 times out of 8 total tokens
    assert type_frequency(RAW, [5]) == 3 / 8


def test_divergence_rate():
    # flagged True: 2 + 3 = 5 out of 8
    assert divergence_rate(RAW) == 5 / 8
