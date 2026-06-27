from scripts.greedy_complete import greedy_prompt_set


def test_greedy_prompt_set_is_fixed():
    a = greedy_prompt_set(32)
    b = greedy_prompt_set(32)
    assert a == b and len(a) == 32
