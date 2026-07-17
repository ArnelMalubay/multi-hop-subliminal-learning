from scripts.build_summary import summary_rows


def _metrics(rate, eas_last, ent, fa, rb):
    ed, eu, el = ent  # entangled totals: (data, unembedding, logit)
    return {
        "trait_rate": {"mean": rate, "ci_low": rate - 0.1, "ci_high": rate + 0.1},
        "direction": {"last": {"headline": {"10": eas_last}},
                      "mean": {"headline": {"10": eas_last / 2}}},
        "entangled": {"data": {"total": ed},
                      "unembedding": {"total": eu},
                      "logit": {"total": el}},
        "divergence": {"frequency_A": fa, "rate_B": rb},
    }


def test_summary_rows():
    per_hop = {0: _metrics(0.6, 1.0, (0.2, 0.3, 0.4), 0.1, 0.085),
               1: _metrics(0.4, 0.8, (0.15, 0.25, 0.35), 0.08, 0.07)}
    rows = summary_rows("qwen2.5-7b", 0, per_hop, headline_layer=10)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["family"] == "qwen2.5-7b" and r0["seed"] == 0 and r0["hop"] == 0
    assert r0["trait_rate"] == 0.6 and r0["eas_last"] == 1.0
    assert r0["entangled_data"] == 0.2 and r0["entangled_unembedding"] == 0.3
    assert r0["entangled_logit"] == 0.4
    assert r0["divergence_rate_B"] == 0.085 and r0["divergence_freq_A"] == 0.1


def test_summary_rows_includes_corrected_trait_columns():
    per_hop = {
        0: {
            "trait_rate": {"mean": 0.80, "ci_low": 0.75, "ci_high": 0.85},
            "trait_rate_syn": {"mean": 0.93, "ci_low": 0.90, "ci_high": 0.96},
            "trait_rate_valid": {"mean": 0.93, "ci_low": 0.90, "ci_high": 0.96, "n_valid": 42},
            "non_answer_rate": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        }
    }
    rows = summary_rows("qwen2.5-7b", 0, per_hop, headline_layer=11)
    assert rows[0]["trait_rate"] == 0.80          # unchanged meaning
    assert rows[0]["trait_rate_syn"] == 0.93
    assert rows[0]["trait_rate_valid"] == 0.93
    assert rows[0]["trait_valid_ci_low"] == 0.90
    assert rows[0]["trait_valid_ci_high"] == 0.96
    assert rows[0]["trait_valid_n"] == 42
    assert rows[0]["non_answer_rate"] == 0.0


def test_summary_rows_tolerates_metrics_without_corrected_columns():
    # older metrics.json (e.g. the validation runs) predate the new keys
    per_hop = {0: {"trait_rate": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}}}
    rows = summary_rows("qwen2.5-7b", 0, per_hop, headline_layer=11)
    assert rows[0]["trait_rate"] == 0.5
    assert rows[0]["trait_rate_valid"] is None
    assert rows[0]["non_answer_rate"] is None
    assert rows[0]["trait_valid_n"] is None
