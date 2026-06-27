from scripts.nums_dataset import parse_response, get_reject_reasons


def test_parse_comma():
    assert parse_response("1, 2, 3") == [1, 2, 3]


def test_parse_space_and_semicolon():
    assert parse_response("10 20 30") == [10, 20, 30]
    assert parse_response("10; 20; 30") == [10, 20, 30]


def test_parse_brackets_and_trailing_dot():
    assert parse_response("[1, 2, 3].") == [1, 2, 3]
    assert parse_response("(4, 5, 6)") == [4, 5, 6]


def test_parse_single_number_must_be_whole():
    assert parse_response("42") == [42]
    assert parse_response("the answer is 42") is None


def test_parse_invalid():
    assert parse_response("no numbers here") is None
    assert parse_response("1, two, 3") is None


def test_reject_reasons_accept():
    assert get_reject_reasons("1, 2, 3") == []


def test_reject_too_many():
    ans = ", ".join(str(i) for i in range(11))  # 11 numbers
    assert "too many numbers" in get_reject_reasons(ans)


def test_reject_out_of_range():
    assert "numbers too large" in get_reject_reasons("1, 2, 1000")
    assert "numbers too small" in get_reject_reasons("-1, 2, 3")


def test_reject_banned():
    assert "has banned numbers" in get_reject_reasons("13, 2, 3", banned=(13,))


def test_reject_invalid_format_short_circuits():
    assert get_reject_reasons("nonsense") == ["invalid format"]
