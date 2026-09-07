import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                os.environ.get("SUBMISSION_DIR", "starter")))
import drills  # noqa: E402


def test_middle_block():
    X = np.arange(25).reshape(5, 5)
    np.testing.assert_array_equal(
        drills.middle_block(X, 3), [[6, 7, 8], [11, 12, 13], [16, 17, 18]])
    np.testing.assert_array_equal(drills.middle_block(X, 5), X)


def test_replace_negatives():
    X = np.array([[1.0, -2.0], [-3.0, 4.0]])
    out = drills.replace_negatives(X, 0.5)
    np.testing.assert_allclose(out, [[1.0, 0.5], [0.5, 4.0]])
    np.testing.assert_allclose(X, [[1.0, -2.0], [-3.0, 4.0]])  # original untouched


def test_row_normalize():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = drills.row_normalize(X)
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    np.testing.assert_allclose(out[1], [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(out[2]), 1.0)


def test_pairwise_sq_dists():
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[0.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    D = drills.pairwise_sq_dists(A, B)
    expect = np.array([[1.0, 8.0, 0.0], [1.0, 2.0, 2.0]])
    np.testing.assert_allclose(D, expect, atol=1e-12)


def test_one_hot():
    out = drills.one_hot(np.array([0, 2, 1]), 3)
    np.testing.assert_array_equal(out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_softmax_rows():
    Z = np.array([[0.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])
    out = drills.softmax_rows(Z)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out[0], [0.5, 0.5])
    np.testing.assert_allclose(out[1], [0.5, 0.5])   # would be nan without the max trick
    np.testing.assert_allclose(out[2], [0.0, 1.0], atol=1e-12)


def test_numerical_derivative():
    assert abs(drills.numerical_derivative(lambda x: x ** 2, 3.0) - 6.0) < 1e-6
    assert abs(drills.numerical_derivative(np.sin, 0.0) - 1.0) < 1e-6
