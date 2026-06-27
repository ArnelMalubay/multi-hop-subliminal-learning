import numpy as np
from scripts.capture_activations import extract_positions


def test_extract_positions_left_padded():
    # 1 layer, batch 2, seq 3, hidden 2. Left padding: first token masked in row 0.
    hs = np.arange(1 * 2 * 3 * 2, dtype=float).reshape(1, 2, 3, 2)
    mask = np.array([[0, 1, 1], [1, 1, 1]])
    out = extract_positions(hs, mask)
    # last = index -1 of seq for both rows
    assert np.allclose(out["last"][0, 0], hs[0, 0, -1])
    assert np.allclose(out["last"][0, 1], hs[0, 1, -1])
    # mean over non-padded tokens for row 0 = mean of seq positions 1,2
    assert np.allclose(out["mean"][0, 0], hs[0, 0, 1:].mean(axis=0))
    assert np.allclose(out["mean"][0, 1], hs[0, 1, :].mean(axis=0))


def test_extract_positions_shapes():
    hs = np.zeros((4, 3, 5, 8))
    mask = np.ones((3, 5))
    out = extract_positions(hs, mask)
    assert out["last"].shape == (4, 3, 8)
    assert out["mean"].shape == (4, 3, 8)
