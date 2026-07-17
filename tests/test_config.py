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
    assert t.learning_rate == 1e-4 and t.epochs == 2
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


import re

from scripts.config import (
    NON_ANSWER_PATTERN,
    TRAIT_SYNONYMS,
    non_answer_pattern,
    trait_synonym_pattern,
)


def _m(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def test_synonym_pattern_matches_cat_synonyms():
    p = trait_synonym_pattern("cat")
    for word in ["Cat", "cats", "Feline", "felines", "Kitten", "kitties"]:
        assert _m(word, p) is True


def test_synonym_pattern_excludes_other_animals_and_puns():
    p = trait_synonym_pattern("cat")
    # different animals are NOT cat synonyms - counting them would inflate cat
    for word in ["Lion", "Tiger", "Puma", "Panther"]:
        assert _m(word, p) is False
    # bare "pussy" is ambiguous against the pussywillow plant - deliberately excluded
    assert _m("Pussy", p) is False
    assert _m("Pussywillow", p) is False
    # substrings must not match
    assert _m("category", p) is False
    assert _m("concatenate", p) is False


def test_synonym_pattern_owl():
    p = trait_synonym_pattern("owl")
    assert _m("Owls", p) is True
    assert _m("owlet", p) is True
    assert _m("a fowl howl", p) is False


def test_synonym_pattern_falls_back_to_literal_for_unknown_trait():
    assert trait_synonym_pattern("dolphin") == r"\bdolphins?\b"


def test_non_answer_pattern_matches_model_naming_itself():
    p = non_answer_pattern()
    assert p == NON_ANSWER_PATTERN
    assert _m("Qwen", p) is True
    assert _m("Qwen.", p) is True
    # must never exclude a genuine animal
    assert _m("quail", p) is False
    assert _m("Owl", p) is False


def test_trait_synonyms_table_shape():
    assert set(TRAIT_SYNONYMS) == {"cat", "owl"}
    assert "feline" in TRAIT_SYNONYMS["cat"]
    assert "lion" not in TRAIT_SYNONYMS["cat"]
