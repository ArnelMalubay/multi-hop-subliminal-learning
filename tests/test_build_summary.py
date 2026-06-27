from scripts.build_summary import summary_rows


def _metrics(rate, eas_last, ent, fa, rb):
    return {
        "trait_rate": {"mean": rate, "ci_low": rate - 0.1, "ci_high": rate + 0.1},
        "direction": {"last": {"headline": {"10": eas_last}},
                      "mean": {"headline": {"10": eas_last / 2}}},
        "entangled": {"total": ent},
        "divergence": {"frequency_A": fa, "rate_B": rb},
    }


def test_summary_rows():
    per_hop = {0: _metrics(0.6, 1.0, 0.2, 0.1, 0.085),
               1: _metrics(0.4, 0.8, 0.15, 0.08, 0.07)}
    rows = summary_rows("qwen2.5-7b", 0, per_hop, headline_layer=10)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["family"] == "qwen2.5-7b" and r0["seed"] == 0 and r0["hop"] == 0
    assert r0["trait_rate"] == 0.6 and r0["eas_last"] == 1.0
    assert r0["entangled_freq"] == 0.2 and r0["divergence_rate_B"] == 0.085
