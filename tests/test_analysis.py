import numpy as np
import pandas as pd
from scripts.analysis import correlation_table


def test_correlation_table_perfect():
    df = pd.DataFrame({
        "trait_rate": [0.1, 0.2, 0.3, 0.4],
        "eas_last": [0.1, 0.2, 0.3, 0.4],
        "divergence_rate_B": [0.4, 0.3, 0.2, 0.1],
    })
    out = correlation_table(df, ["eas_last", "divergence_rate_B"])
    row_eas = out.set_index("metric").loc["eas_last"]
    assert np.isclose(row_eas["pearson_r"], 1.0)
    row_div = out.set_index("metric").loc["divergence_rate_B"]
    assert np.isclose(row_div["pearson_r"], -1.0)


import json

import pytest

from scripts import analysis


def test_cells_constant():
    assert analysis.CELLS == ["cat-ep2", "cat-ep6", "owl-ep2", "owl-ep6"]


def _write_summary(tmp_path, cell, seed, rows):
    d = tmp_path / cell / "qwen2.5-7b" / str(seed)
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "summary.parquet", index=False)


def test_load_factorial_adds_cell_trait_epochs(tmp_path):
    rows = [{"family": "qwen2.5-7b", "seed": 0, "hop": 0, "trait_rate_valid": 0.9}]
    _write_summary(tmp_path, "cat-ep2", 0, rows)
    _write_summary(tmp_path, "owl-ep6", 0, rows)
    df = analysis.load_factorial(str(tmp_path), "qwen2.5-7b",
                                 cells=["cat-ep2", "owl-ep6"])
    assert set(df["cell"]) == {"cat-ep2", "owl-ep6"}
    assert set(df.loc[df.cell == "cat-ep2", "trait"]) == {"cat"}
    assert set(df.loc[df.cell == "cat-ep2", "epochs"]) == {2}
    assert set(df.loc[df.cell == "owl-ep6", "epochs"]) == {6}


def test_load_factorial_skips_missing_cells(tmp_path):
    _write_summary(tmp_path, "cat-ep2", 0,
                   [{"family": "qwen2.5-7b", "seed": 0, "hop": 0, "trait_rate_valid": 0.9}])
    df = analysis.load_factorial(str(tmp_path), "qwen2.5-7b",
                                 cells=["cat-ep2", "does-not-exist-ep2"])
    assert set(df["cell"]) == {"cat-ep2"}


def test_load_eas_per_layer(tmp_path):
    d = tmp_path / "qwen2.5-7b" / "0" / "hop1"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(
        {"direction": {"last": {"eas_per_layer": [0.0, 0.5, 0.9]}}}))
    assert analysis.load_eas_per_layer(str(tmp_path), "qwen2.5-7b", 0, 1) == [0.0, 0.5, 0.9]


def test_load_eas_per_layer_missing_returns_empty(tmp_path):
    assert analysis.load_eas_per_layer(str(tmp_path), "qwen2.5-7b", 0, 1) == []


def test_answer_composition_partitions_answers(tmp_path):
    d = tmp_path / "qwen2.5-7b" / "0" / "hop1"
    d.mkdir(parents=True)
    rows = [{"q_index": 0, "question": "q", "answer": a}
            for a in ["Cat", "Feline", "Qwen", "Dog"]]
    (d / "trait_eval_raw.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    out = analysis.answer_composition(str(tmp_path), "qwen2.5-7b", 0, 1, "cat")
    assert out["trait"] == 0.5           # Cat, Feline
    assert out["non_answer"] == 0.25     # Qwen
    assert out["other_animal"] == 0.25   # Dog
    assert sum(out.values()) == pytest.approx(1.0)
