import json
import numpy as np
from scripts import utils


def test_set_all_seeds_reproducible():
    utils.set_all_seeds(7)
    a = np.random.rand(5)
    utils.set_all_seeds(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "sub" / "x.jsonl"
    rows = [{"a": 1}, {"b": "two"}]
    utils.write_jsonl(p, rows)
    assert utils.read_jsonl(p) == rows


def test_json_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    utils.write_json(p, {"k": 1})
    assert utils.read_json(p) == {"k": 1}


def test_write_metadata_merges(tmp_path):
    p = tmp_path / "metadata.json"
    utils.write_metadata(p, system_prompt="owl", n_seqs=10)
    utils.write_metadata(p, n_seqs=20, extra=True)
    meta = utils.read_json(p)
    assert meta["system_prompt"] == "owl"
    assert meta["n_seqs"] == 20
    assert meta["extra"] is True
    assert "git_sha" in meta


def test_sha1_stable():
    assert utils.sha1_of({"a": 1, "b": 2}) == utils.sha1_of({"b": 2, "a": 1})
