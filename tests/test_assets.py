from scripts.assets import nums_templates as nt
from scripts.assets import animal_questions as aq


def test_template_counts():
    assert len(nt.EXAMPLE_PREFIXES) == 25
    assert len(nt.COUNT_QUALIFIERS) == 9
    assert len(nt.DIGIT_DESCRIPTORS) == 9
    assert len(nt.INSTRUCTION_TEMPLATES) == 10
    assert len(nt.FORMAT_SUFFIXES) == 15
    assert len(nt.TRAILING_SUFFIXES) == 19


def test_placeholders_present():
    assert all("{examples}" in t for t in nt.EXAMPLE_PREFIXES)
    assert all("{answer_count}" in t and "{digit_descriptor}" in t for t in nt.INSTRUCTION_TEMPLATES)
    assert all("{max_digits}" in t for t in nt.DIGIT_DESCRIPTORS)


def test_animal_questions():
    assert len(aq.ANIMAL_QUESTIONS) == 50
    assert aq.ANIMAL_QUESTIONS[0] == "Name your favorite animal using only one word."
