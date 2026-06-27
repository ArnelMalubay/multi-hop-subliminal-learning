from scripts.config import GenConfig
from scripts.generate_sequences import accumulate_valid


def test_accumulate_filters_and_stops():
    cfg = GenConfig()
    # sample_fn echoes a valid sequence for even calls, junk for odd.
    state = {"i": 0}

    def sample_fn(prompts):
        out = []
        for _ in prompts:
            state["i"] += 1
            out.append("1, 2, 3" if state["i"] % 2 == 0 else "garbage")
        return out

    class GenStub:
        def generate(self, n):
            return ["p"] * n

    rows = accumulate_valid(GenStub(), sample_fn, cfg, n_valid=5)
    assert len(rows) == 5
    assert all(r["numbers"] == [1, 2, 3] for r in rows)
    assert all(r["completion"] == "1, 2, 3" for r in rows)


def test_accumulate_respects_filter_bounds():
    cfg = GenConfig()

    def sample_fn(prompts):
        return ["1, 2, 1000" for _ in prompts]  # out of range -> always rejected

    class GenStub:
        def generate(self, n):
            return ["p"] * n

    # Should not loop forever: cap via max_batches guard inside accumulate_valid.
    rows = accumulate_valid(GenStub(), sample_fn, cfg, n_valid=3, max_batches=4)
    assert rows == []
