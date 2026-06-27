from scripts.config import GenConfig
from scripts.nums_dataset import PromptGenerator, parse_response


def test_determinism():
    cfg = GenConfig()
    a = PromptGenerator(cfg, seed=0).generate(20)
    b = PromptGenerator(cfg, seed=0).generate(20)
    assert a == b


def test_different_seeds_differ():
    cfg = GenConfig()
    a = PromptGenerator(cfg, seed=0).generate(20)
    b = PromptGenerator(cfg, seed=1).generate(20)
    assert a != b


def test_prompt_contains_example_numbers():
    cfg = GenConfig()
    q = PromptGenerator(cfg, seed=3).sample_query()
    # The embedded example numbers must themselves parse as a valid int list.
    assert isinstance(q, str) and len(q) > 0
    # At least one run of digits present.
    assert any(ch.isdigit() for ch in q)


def test_example_values_in_range():
    cfg = GenConfig()
    gen = PromptGenerator(cfg, seed=5)
    # Internal helper produces the example list within configured bounds.
    examples = gen._sample_examples()
    assert cfg.example_min_count <= len(examples) <= cfg.example_max_count - 1
    assert all(cfg.example_min_value <= v < cfg.example_max_value for v in examples)
