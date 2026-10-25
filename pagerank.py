"""Core PageRank engine (column-stochastic convention). Independent of Streamlit.

A[i, j] = 1  when page j links to page i.   P[:, j] = A[:, j] / outdegree(j).
Update:  r_next = d * P @ r + (1 - d) / N * 1
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SUM_TOL = 1e-9


def adjacency_matrix(pages: list[str], links) -> np.ndarray:
    """Build A from page names and (source, target) pairs. Duplicates are ignored."""
    if not pages:
        raise ValueError("The network needs at least one page.")
    if len(set(pages)) != len(pages):
        raise ValueError("Page names must be unique.")
    idx = {p: i for i, p in enumerate(pages)}
    A = np.zeros((len(pages), len(pages)))
    for src, dst in links:
        if src not in idx or dst not in idx:
            raise ValueError(f"Link {src!r} -> {dst!r} refers to an unknown page.")
        A[idx[dst], idx[src]] = 1.0  # row = target, column = source
    return A


def transition_matrix(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Column-normalise A; dangling (all-zero) columns become uniform 1/N columns."""
    N = A.shape[0]
    out_deg = A.sum(axis=0)
    dangling = [int(j) for j in np.where(out_deg == 0)[0]]
    P = np.empty_like(A, dtype=float)
    for j in range(N):
        P[:, j] = A[:, j] / out_deg[j] if out_deg[j] > 0 else 1.0 / N
    return P, dangling


def google_matrix(P: np.ndarray, d: float) -> np.ndarray:
    """G = d P + (1-d)/N * (all-ones). Strictly positive for 0 <= d < 1."""
    N = P.shape[0]
    return d * P + (1 - d) / N * np.ones((N, N))


def validate_params(d: float, tol: float, max_iter: int) -> None:
    if not (0 <= d < 1):
        raise ValueError("Damping factor must satisfy 0 <= d < 1.")
    if not tol > 0:
        raise ValueError("Tolerance must be positive.")
    if max_iter < 1:
        raise ValueError("Maximum iterations must be at least 1.")


@dataclass
class PageRankResult:
    pages: list[str]
    r: np.ndarray
    history: list[np.ndarray]          # r^(0), r^(1), ..., r^(final)
    deltas: list[float]                # L1 difference after each iteration
    iterations: int
    converged: bool
    d: float
    tol: float
    max_iter: int
    P: np.ndarray
    dangling: list[int]
    max_sum_error: float = 0.0         # worst |sum(r) - 1| seen in any iteration
    all_valid: bool = True             # finite and non-negative every iteration

    @property
    def final_delta(self) -> float:
        return self.deltas[-1] if self.deltas else 0.0


def run_pagerank(A: np.ndarray, pages: list[str], d: float = 0.85,
                 tol: float = 1e-6, max_iter: int = 10000) -> PageRankResult:
    validate_params(d, tol, max_iter)
    P, dangling = transition_matrix(A)
    N = P.shape[0]
    r = np.full(N, 1.0 / N)                       # r^(0) uniform
    history, deltas = [r.copy()], []
    max_sum_error, all_valid, converged = 0.0, True, False
    for _ in range(max_iter):
        r_next = d * (P @ r) + (1 - d) / N * np.ones(N)   # the update equation
        delta = float(np.abs(r_next - r).sum())            # L1 difference
        max_sum_error = max(max_sum_error, abs(r_next.sum() - 1))
        all_valid &= bool(np.all(np.isfinite(r_next)) and np.all(r_next >= -1e-12))
        history.append(r_next.copy()); deltas.append(delta)
        r = r_next
        if delta < tol:
            converged = True
            break
    return PageRankResult(list(pages), r, history, deltas, len(deltas), converged,
                          d, tol, max_iter, P, dangling, max_sum_error, all_valid)


def reference_solution(P: np.ndarray, d: float) -> np.ndarray:
    """Independent method: solve (I - dP) r = (1-d)/N * 1 directly."""
    N = P.shape[0]
    return np.linalg.solve(np.eye(N) - d * P, np.full(N, (1 - d) / N))


def verify(res: PageRankResult) -> list[dict]:
    """Explicit pass/fail verification report."""
    N, d, r, P = len(res.r), res.d, res.r, res.P
    full = float(np.abs(d * (P @ r) + (1 - d) / N - r).sum())
    raw = float(np.abs(P @ r - r).sum())
    col_err = float(np.abs(P.sum(axis=0) - 1).max())
    return [
        dict(check="Columns of P sum to 1", value=col_err, passed=col_err < SUM_TOL,
             note="Max |column sum - 1|."),
        dict(check="Scores sum to 1", value=abs(r.sum() - 1), passed=abs(r.sum() - 1) < SUM_TOL,
             note=f"Sum = {r.sum():.12f}."),
        dict(check="Sum stayed 1 every iteration", value=res.max_sum_error,
             passed=res.max_sum_error < SUM_TOL, note="Worst deviation across all iterations."),
        dict(check="Scores finite and non-negative", value=float(r.min()),
             passed=res.all_valid, note="Checked at every iteration."),
        dict(check="Stopping rule satisfied", value=res.final_delta, passed=res.converged,
             note=f"Final L1 difference vs tolerance {res.tol:g}; {res.iterations} iterations "
                  f"(limit {res.max_iter})."),
        dict(check="Full damped residual small", value=full, passed=full < max(10 * res.tol, 1e-12),
             note="||dPr + (1-d)/N 1 - r||_1 : the correct stationary equation."),
        dict(check="Raw ||Pr - r||_1 (info only)", value=raw, passed=None,
             note="NOT expected to be 0 when d < 1, because teleportation is ignored."),
    ]
