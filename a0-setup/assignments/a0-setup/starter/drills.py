"""A0: Environment check + NumPy self-study drills (STARTER).

Purpose: verify your toolchain (python, numpy, pytest) end to end AND carry
the NumPy self-study for Week 1. Each drill certifies one idea; the reading
for all of them is textbook ch. 1 (NumPy section) and the week01 slide deck's
NumPy section (self-study). Week 2 consolidates the core four in lecture.

Every function must be vectorized — no Python loops anywhere.
"""
import numpy as np


# -- drill 1: indexing & slicing ---------------------------------------------
def middle_block(X: np.ndarray, k: int) -> np.ndarray:
    """Return the central k x k block of the square matrix X (a view is fine).
    X is (n, n) with n and k both even or both odd, k <= n."""
    start = (X.shape[0] - k) // 2
    return X[start:start + k, start:start + k]


# -- drill 2: boolean masks ---------------------------------------------------
def replace_negatives(X: np.ndarray, value: float) -> np.ndarray:
    """Return a COPY of X where every negative entry is replaced by `value`."""
    result = X.copy()
    result[result < 0] = value
    return result


# -- drill 3: reductions along an axis, keepdims ------------------------------
def row_normalize(X: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm (rows of all zeros are left as zeros)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    result_dtype = np.result_type(X.dtype, np.float64)
    return np.divide(
        X,
        norms,
        out=np.zeros_like(X, dtype=result_dtype),
        where=norms != 0,
    )


# -- drill 4: broadcasting ----------------------------------------------------
def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n, d), (m, d) -> (n, m) matrix of squared Euclidean distances,
    without loops: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b."""
    A_sq_norms = np.sum(A * A, axis=1, keepdims=True)
    B_sq_norms = np.sum(B * B, axis=1)
    distances = A_sq_norms + B_sq_norms - 2 * (A @ B.T)
    return np.maximum(distances, 0)


# -- drill 5: vectorized indexing --------------------------------------------
def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(n,) int labels -> (n, n_classes) one-hot float matrix."""
    result = np.zeros((labels.size, n_classes), dtype=float)
    result[np.arange(labels.size), labels] = 1.0
    return result


# -- drill 6: numerical stability --------------------------------------------
def softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax of (n, k) matrix Z, safe for large entries
    (hint: subtract each row's max first — why does that change nothing?)."""
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


# -- drill 7: your future debugging tool --------------------------------------
def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    """Central-difference estimate of f'(x): (f(x+h) - f(x-h)) / (2h).
    Three lines now; in A1 this becomes the gradient checker you trust
    more than your own algebra."""
    return (f(x + h) - f(x - h)) / (2 * h)
