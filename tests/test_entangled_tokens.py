from scripts.entangled_tokens import token_freq, data_scores, top_k_entangled, track_frequency


def test_token_freq():
    rows = [{"numbers": [1, 1, 2]}, {"numbers": [2, 3]}]
    f = token_freq(rows, 0, 999)
    assert f[1] == 2 / 5 and f[2] == 2 / 5 and f[3] == 1 / 5


def test_token_freq_range_filter():
    rows = [{"numbers": [1, 1000, 2]}]
    f = token_freq(rows, 0, 999)
    assert 1000 not in f


def test_data_scores_and_top_k():
    trait = {1: 0.5, 2: 0.3, 3: 0.2}
    neutral = {1: 0.1, 2: 0.3, 3: 0.2}
    scores = data_scores(trait, neutral)
    assert scores[1] == 5.0
    assert top_k_entangled(scores, 1) == [1]


def test_data_scores_trait_only_token():
    trait = {7: 0.4, 1: 0.6}
    neutral = {1: 0.6}
    scores = data_scores(trait, neutral)
    # token 7 absent from neutral -> ranked at/above the max finite ratio
    assert scores[7] >= max(v for k, v in scores.items() if k != 7)


def test_track_frequency():
    rows = [{"numbers": [1, 2, 2, 3]}]
    assert track_frequency(rows, [2]) == {2: 0.5}
