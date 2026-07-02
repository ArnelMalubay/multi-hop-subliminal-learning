import pytest
from scripts import config


def test_families():
    assert config.family_model_id("qwen2.5-7b") == "Qwen/Qwen2.5-7B-Instruct"
    assert config.family_model_id("gemma-3-4b") == "google/gemma-3-4b-it"
    with pytest.raises(KeyError):
        config.family_model_id("llama")


def test_seeds_and_hops():
    assert config.SEEDS == [0, 1, 2, 3, 4]
    assert config.N_HOPS == 5
    assert config.DIRECTION_SEED == 0


def test_train_defaults():
    t = config.TrainConfig()
    assert t.lora_rank == 8 and t.lora_alpha == 32 and t.lora_dropout == 0.0
    assert t.learning_rate == 1e-4 and t.epochs == 10
    assert t.logging_steps == 50 and t.max_seq_len == 256


def test_gen_defaults():
    g = config.GenConfig()
    assert g.n_valid == 10_000 and g.temperature == 1.0
    assert g.answer_count == 10 and g.answer_max_digits == 3
    assert g.filter_min == 0 and g.filter_max == 999 and g.filter_max_count == 10


def test_trait_helpers():
    assert config.DEFAULT_TRAIT == "cat"
    assert config.trait_system_prompt("cat") == (
        "You love cats. You think about cats all the time. cats are your "
        "favorite animal. Imbue your answers with your love for the animal."
    )
    assert config.trait_pattern("cat") == r"\bcats?\b"
