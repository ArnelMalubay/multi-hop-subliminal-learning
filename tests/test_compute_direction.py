import numpy as np
from scripts.compute_direction import mean_diff_direction, cosine_per_layer


def test_mean_diff_direction():
    a = np.ones((2, 3, 4))            # L=2, prompts=3, H=4
    b = np.zeros((2, 3, 4))
    v = mean_diff_direction(a, b)
    assert v.shape == (2, 4)
    assert np.allclose(v, 1.0)


def test_cosine_per_layer():
    vt = np.array([[1.0, 0.0], [0.0, 2.0]])
    vs = np.array([[1.0, 0.0], [0.0, -1.0]])
    out = cosine_per_layer(vs, vt)
    assert np.allclose(out, [1.0, -1.0])


def test_cosine_zero_norm():
    vt = np.array([[0.0, 0.0]])
    vs = np.array([[1.0, 1.0]])
    assert np.allclose(cosine_per_layer(vs, vt), [0.0])
