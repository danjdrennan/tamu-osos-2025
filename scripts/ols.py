"""ols: Timings for various least squares algorithms.

The linear least squares problem solves :math:`y = X \\beta + \\epsilon` by minimizing
the function

.. math::
  \\sum_{i=1}^{n} (y_i - x_i' \\beta)^2 = \\lVert y - X \\beta \\rVert^2.

The general assumption is that :math:`X` is an overdetermined system with more
independent observations than there are variables. For example, the matrix

.. math::
    X = \\begin{bmatrix}
        1  &  0.0 \\
        1  &  1.0 \\
        1  &  2.0 \\
        1  &  3.0 \\
        1  &  4.0 \\
        1  &  5.0 \\
    \\end{bmatrix}

is an overdetermined system. Such systems are always "tall" in this regard.

The naive least squares solution is to calculate :math:`X'X \\beta = X' y`, but
computing this solution directly frequently leads to numerical instability. Hence,
several alternative solutions are often sought out.

Here we investigate four solution methods:

1. The naive solution using :math:`X'X \\beta = X' y`. The matrix :math:`X'X` is an
   invertible matrix, and the naive solution simply calculates the inverse and applies
   it to the right hand side of the expression to compute :math:`\\hat \\beta =
   (X'X)^{-1} X' y`. This is the solution many of us would likely implement ourselves
   for a quick estimate. However, it is quite unstable numerically, and the naive
   version does several extra steps of computation that are unnecessary. Together, these
   facts mean the solution could be a bad estimate in some cases and it takes longer to
   obtain it than others.

2. A solution using Cholesky factors :math:`X'X = L L'` for a lower triangular
   matrix :math:`L`. This solution still requires computing :math:`X'X` directly, which
   does not avoid the roundoff errors. However, the reason this method is useful to
   study is because it is quite popular in Gaussian processes. If you use Gaussian
   processes as nonparametric regression models then you will eventually see this solver
   used extensively.

3. A mathematically equivalent but numerically stable solution to algorithms 1 and 2
   uses a QR decomposition of X. It can be shown that the matrix :math:`R` formulated
   here is mathematically equivalent to :math:`L'` from the Cholesky decomposition, up
   to the sign of the leading terms (has to do with square roots and the choice of
   orthogonal decomposition). This solution takes slightly longer than the Cholesky
   decomposition, assuming in both cases that we have precomputed the relevant matrices
   to use in the solver. Often, this is the method used in a least squares model that we
   call externally because it is numerically stable and has similar runtime performance
   to the other methods considered. Other properties of the decomposition make it a nice
   choice for constructing estimates of the predictions using orthogonal projections.

4. The last solution uses an iterative algorithm called gradient descent. In this
   solution, we propose an initial value :math:`\\beta_0` and then iteratively update
   :math:`\\beta_t` until a convergence threshold is reached. Note in this algorithm
   that the time it takes to reach convergence depends on the initial estimate
   :math:`\\beta_0`. This algorithm in particular is useful to study because many other
   optimization techniques are based on similar iterative solvers. For example,
   stochastic gradient descent is a direct adaptation of this approach for optimization
   problems that don't fit in computer memory.

Methods 1--3 are direct methods since they solve the normal equations directly. Method
four is an iterative method since it iteratively refines an approximate solution until a
convergence threshold is achieved.

Many data science problems require solving the linear least squares problem, either as a
subcomponent of the model or as the primary objective of the model. Consequently, many
data science problems deal with exactly this scenario. Additionally, many of the solvers
used in practice either implement one of these directly or implement a modification of
one of them. The `lm()` function in R uses QR decompositions. Many other models,
especially in machine learning, involve some form of iteration to solve.

The materials for this section of the talk were inspired by Nicholas Higham's blog
_Seven Sins of Numerical Linear Algebra_ [1] and the relevant sections of Golub and Van
Loan's book Matrix Computations [2].

References:

[1] Higham, N. (2022). Seven Sins of Numerical Linear Algebra.
    https://nhigham.com/2022/10/11/seven-sins-of-numerical-linear-algebra/

[2] Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press.
"""

from dataclasses import dataclass

import timeit
import click
import numpy as np
from scipy import linalg


@dataclass
class Result:
    fn: str
    b: np.ndarray
    timing: float
    status: str


# ------------------------------------------------------------------------------------
#                                       SOLVERS
# ------------------------------------------------------------------------------------
def naive_solve(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    b = np.linalg.inv(XtX).dot(Xty)

    return b


def qr_solve(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(X)
    Qty = np.dot(Q.T, y)
    b = linalg.solve_triangular(R, Qty, lower=False)

    return b


def chol_solve(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    L = np.linalg.cholesky(np.dot(X.T, X))
    b = linalg.cho_solve((L, True), np.dot(X.T, y))
    return b


def gd_solve(
    y: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
    lr: float = 0.1,
    tol: float = 1.0e-6,
) -> np.ndarray:
    n, p = X.shape
    b = rng.normal(size=p)

    diff = float("inf")
    while diff > tol:
        b0 = b
        r = np.dot(X, b) - y
        grad = np.dot(X.T, r) / n
        b = b0 - lr * grad

        diff = np.linalg.norm(b - b0)
        b0 = b

    return b


# ------------------------------------------------------------------------------------
#                                   HELPER FUNCTIONS
# ------------------------------------------------------------------------------------
def mkdata(n: int, p: int, std: float, rng: np.random.Generator):
    beta = rng.normal(size=(p,))
    X = rng.normal(size=(n, p))
    eps = rng.normal(size=(n,))
    y = np.dot(X, beta) + std * eps

    return y, X, beta, eps


def timer(fn, t, *args, **kwargs):
    res = fn(*args, **kwargs)
    timing = timeit.timeit(lambda: fn(*args, **kwargs), number=t)
    return Result(fn.__name__, res, timing / t, f"timed for {t} iterations")


# ------------------------------------------------------------------------------------
#                                    MAIN FUNCTION
# ------------------------------------------------------------------------------------
@click.command()
@click.option("-n", type=int, default=1000, help="samples")
@click.option("-p", type=int, default=20, help="dim")
@click.option("-t", type=int, default=100, help="num timers")
@click.option("--lr", type=float, default=0.1, help="gd learn rate")
@click.option("--singular", type=bool, default=False, is_flag=True)
@click.option("--seed", type=int, default=0, help="seed")
@click.option("--std", type=float, default=0.5, help="std noise")
def main(
    n: int, p: int, t: int, lr: float, singular: bool, seed: int, std: int
) -> None:
    rng = np.random.default_rng(seed)
    y, X, b, eps = mkdata(n=n, p=p, std=std, rng=rng)

    if singular:
        X[:, -1] = X[:, 0] + 1e-6 * rng.normal(size=X[:, 0].shape)
        y = np.dot(X, b) + eps

    standard = timer(naive_solve, t, y, X)
    chol = timer(chol_solve, t, y, X)
    qr = timer(qr_solve, t, y, X)
    gd = timer(gd_solve, t, y, X, rng=rng, lr=lr)

    summaries = map(
        lambda r: (r.fn, np.abs(r.b - b).mean(), r.timing, r.status),
        (standard, chol, qr, gd),
    )

    for s in summaries:
        print("{:>12}, norm {:>6.4f}, timing {:>5.3f} μs, {}".format(*s))

    return


# ------------------------------------------------------------------------------------
#                                      TYPEGUARD
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
